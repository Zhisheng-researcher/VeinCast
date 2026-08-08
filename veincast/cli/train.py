from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import resource
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.checkpoint import checkpoint as activation_checkpoint

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None

from ..config import load_config, resolve_config_paths, save_config
from ..data import FieldNormalizer, VeinCastERA5Dataset
from ..losses import VeinCastForecastLoss
from ..metrics import MetricAccumulator
from ..model import VeinCast
from ..variables import VariableRegistry


warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cpu\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
    category=FutureWarning,
)


@dataclass(frozen=True)
class DistributedSettings:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def distributed_settings_from_env() -> DistributedSettings:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    return DistributedSettings(enabled, rank, world_size, local_rank, backend)


def setup_distributed() -> tuple[DistributedSettings, torch.device]:
    settings = distributed_settings_from_env()
    if settings.enabled:
        if torch.cuda.is_available():
            torch.cuda.set_device(settings.local_rank)
            device = torch.device("cuda", settings.local_rank)
        else:
            device = torch.device("cpu")
        if not dist.is_initialized():
            dist.init_process_group(backend=settings.backend, init_method="env://")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return settings, device


def cleanup_distributed(settings: DistributedSettings) -> None:
    if settings.enabled and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def reduce_metrics(
    metrics: Dict[str, float],
    device: torch.device,
    settings: DistributedSettings,
) -> Dict[str, float]:
    if not settings.enabled:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor(
        [float(metrics[key]) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= settings.world_size
    return {key: float(value.item()) for key, value in zip(keys, values)}


def any_rank_true(value: bool, device: torch.device, settings: DistributedSettings | None) -> bool:
    if settings is None or not settings.enabled or not dist.is_initialized():
        return bool(value)
    flag = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def metric_to_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().mean().item())
    return float(value)


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    values = tensor.detach().float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def per_field_diagnostic(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    field_names: Tuple[str, ...],
    top_k: int,
) -> Dict[str, object]:
    with torch.no_grad():
        mask = valid_mask.float()
        squared = (prediction.detach().float() - target.detach().float()).square() * mask
        counts = mask.sum(dim=(0, 2, 3)).clamp_min(1.0)
        mse = squared.sum(dim=(0, 2, 3)) / counts
        top_k = max(1, min(int(top_k), int(mse.numel())))
        values, indices = torch.topk(mse, k=top_k)
        fields = []
        prediction_float = prediction.detach().float()
        target_float = target.detach().float()
        for value, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
            field_name = field_names[index] if index < len(field_names) else str(index)
            fields.append(
                {
                    "field": field_name,
                    "mse": float(value),
                    "prediction": tensor_stats(prediction_float[:, index]),
                    "target": tensor_stats(target_float[:, index]),
                }
            )
        return {
            "prediction": tensor_stats(prediction_float),
            "target": tensor_stats(target_float),
            "top_fields": fields,
        }


def resolve_amp_dtype(device: torch.device, amp_dtype: str | None = None) -> torch.dtype:
    value = str(amp_dtype or "float16").lower()
    if value in {"bf16", "bfloat16"}:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")


def autocast_context(device: torch.device, enabled: bool, amp_dtype: str | None = None):
    if not enabled or device.type != "cuda":
        return nullcontext()
    dtype = resolve_amp_dtype(device, amp_dtype)
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(
    device: torch.device,
    enabled: bool,
    amp_dtype: str | None = None,
) -> torch.amp.GradScaler:
    dtype = resolve_amp_dtype(device, amp_dtype) if enabled and device.type == "cuda" else None
    return torch.amp.GradScaler(
        "cuda",
        enabled=enabled and device.type == "cuda" and dtype == torch.float16,
    )


def build_optimizer(
    model: torch.nn.Module,
    training_config: Dict[str, object],
) -> tuple[torch.optim.Optimizer, Dict[str, object]]:
    learning_rate = float(training_config["learning_rate"])
    weight_decay = float(training_config["weight_decay"])
    sensitive_multiplier = float(
        training_config.get("sensitive_lr_multiplier", 1.0)
    )
    sensitive_patterns = tuple(
        str(pattern)
        for pattern in training_config.get(
            "sensitive_parameter_patterns",
            (
                "recovery.",
                "fusion_feedback.gate",
                "fusion_query_reader.gate",
                "field_to_fusion.centrality_bias",
            ),
        )
    )

    def is_sensitive(name: str) -> bool:
        return any(pattern in name for pattern in sensitive_patterns)

    def use_weight_decay(name: str, parameter: torch.nn.Parameter) -> bool:
        lowered = name.lower()
        if parameter.ndim <= 1 or lowered.endswith(".bias"):
            return False
        no_decay_tokens = (
            "norm",
            "embedding",
            "position_bias",
            "relative_table",
            "latitude_bias",
            "variable_affine",
            "latents",
        )
        return not any(token in lowered for token in no_decay_tokens)

    groups: Dict[tuple[bool, bool], list[torch.nn.Parameter]] = {
        (False, True): [],
        (False, False): [],
        (True, True): [],
        (True, False): [],
    }
    counts: Dict[str, int] = {
        "base_decay": 0,
        "base_no_decay": 0,
        "sensitive_decay": 0,
        "sensitive_no_decay": 0,
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        sensitive = is_sensitive(name)
        decay = use_weight_decay(name, parameter)
        groups[(sensitive, decay)].append(parameter)
        label = (
            "sensitive_decay" if sensitive and decay else
            "sensitive_no_decay" if sensitive else
            "base_decay" if decay else
            "base_no_decay"
        )
        counts[label] += int(parameter.numel())

    optimizer_groups = []
    for sensitive, decay in ((False, True), (False, False), (True, True), (True, False)):
        params = groups[(sensitive, decay)]
        if not params:
            continue
        optimizer_groups.append(
            {
                "params": params,
                "lr": learning_rate * (sensitive_multiplier if sensitive else 1.0),
                "weight_decay": weight_decay if decay else 0.0,
            }
        )
    summary = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "sensitive_lr_multiplier": sensitive_multiplier,
        "sensitive_patterns": list(sensitive_patterns),
        "parameter_counts": counts,
    }
    return torch.optim.AdamW(optimizer_groups), summary


def metric_tensor_to_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().item())
    return float(value)


def compute_gradient_norm(model: torch.nn.Module) -> float:
    squared_norm = None
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().norm(2).square()
        squared_norm = value if squared_norm is None else squared_norm + value
    if squared_norm is None:
        return 0.0
    return float(squared_norm.sqrt().item())


def log_memory_stage(
    label: str,
    device: torch.device,
    settings: DistributedSettings,
) -> None:
    if not bool(os.environ.get("WEATHER_TRAIN_DEBUG_MEMORY", "0")):
        return
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    payload = {
        "stage": label,
        "rank": settings.rank,
        "local_rank": settings.local_rank,
        "rss_mb": round(rss_mb, 1),
    }
    if device.type == "cuda":
        payload.update(
            {
                "cuda_allocated_mb": round(
                    torch.cuda.memory_allocated(device) / (1024.0**2), 1
                ),
                "cuda_reserved_mb": round(
                    torch.cuda.memory_reserved(device) / (1024.0**2), 1
                ),
                "cuda_max_allocated_mb": round(
                    torch.cuda.max_memory_allocated(device) / (1024.0**2), 1
                ),
            }
        )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def clear_cuda_cache(
    label: str,
    device: torch.device,
    settings: DistributedSettings,
) -> None:
    if device.type != "cuda":
        return
    torch.cuda.empty_cache()
    log_memory_stage(label, device, settings)


def make_checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    best_valid: float,
    config: Dict[str, object],
    registry: VariableRegistry,
    normalizer: FieldNormalizer,
) -> Dict[str, object]:
    return {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_valid": float(best_valid),
        "config": config,
        "registry": registry.to_dict(),
        "normalizer": {
            "field_keys": list(normalizer.field_keys),
            "mean": normalizer.mean.tolist(),
            "std": normalizer.std.tolist(),
        },
    }


def restore_training_state(
    checkpoint: Dict[str, object],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
) -> Tuple[int, float]:
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1, float(
        checkpoint.get("best_valid", float("inf"))
    )


def _load_state_dict_with_module_fallback(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> None:
    try:
        model.load_state_dict(state_dict, strict=strict)
        return
    except RuntimeError:
        pass

    stripped = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(stripped, strict=strict)


def initialize_model_weights(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    strict: bool = True,
) -> None:
    """Load only model weights for a new training stage.

    This is intentionally separate from --resume: staged rollout finetuning
    should usually reset optimizer, scheduler and GradScaler state.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    _load_state_dict_with_module_fallback(model, state_dict, strict=strict)


def forward_model_step(
    model: torch.nn.Module,
    state: torch.Tensor,
    static: torch.Tensor,
    input_present: torch.Tensor,
    lead_hours: torch.Tensor,
    calendar_features: torch.Tensor | None,
    use_activation_checkpoint: bool,
    preserve_rng_state: bool,
) -> Dict[str, object]:
    """Run one forecast step, optionally recomputing activations in backward."""

    checkpoint_enabled = use_activation_checkpoint and torch.is_grad_enabled()
    if not checkpoint_enabled:
        calendar_kwargs = (
            {"calendar_features": calendar_features}
            if calendar_features is not None
            else {}
        )
        return model(
            state,
            static,
            input_present=input_present,
            lead_hours=lead_hours,
            **calendar_kwargs,
        )

    if calendar_features is None:
        def run(state_tensor, static_tensor, present_tensor, lead_tensor):
            output = model(
                state_tensor,
                static_tensor,
                input_present=present_tensor,
                lead_hours=lead_tensor,
            )
            return output["prediction"]

        prediction = activation_checkpoint(
            run,
            state,
            static,
            input_present,
            lead_hours,
            use_reentrant=False,
            preserve_rng_state=preserve_rng_state,
        )
    else:
        def run(state_tensor, static_tensor, present_tensor, lead_tensor, calendar_tensor):
            output = model(
                state_tensor,
                static_tensor,
                input_present=present_tensor,
                lead_hours=lead_tensor,
                calendar_features=calendar_tensor,
            )
            return output["prediction"]

        prediction = activation_checkpoint(
            run,
            state,
            static,
            input_present,
            lead_hours,
            calendar_features,
            use_reentrant=False,
            preserve_rng_state=preserve_rng_state,
        )

    return {"prediction": prediction, "aux": {}}


def build_datasets(
    config: Dict[str, object],
    registry: VariableRegistry,
    settings: DistributedSettings | None = None,
):
    data_config = config["data"]
    stats_path = Path(data_config["stats_path"])
    distributed = settings is not None and settings.enabled

    if stats_path.exists():
        normalizer = FieldNormalizer.load(stats_path)
    elif distributed and not settings.is_main:
        while not stats_path.exists():
            time.sleep(30.0)
        normalizer = FieldNormalizer.load(stats_path)
    else:
        normalizer = FieldNormalizer.fit_from_path(
            data_config["dynamic_path"],
            registry,
            data_config["train_start"],
            data_config["train_end"],
        )
        normalizer.save(stats_path)
    if distributed:
        dist.barrier()

    variable_minimum_std = data_config.get("variable_minimum_normalization_std", {})
    if variable_minimum_std is None:
        variable_minimum_std = {}
    normalizer = normalizer.with_std_floor(
        registry=registry,
        minimum_std=float(data_config.get("minimum_normalization_std", 0.0)),
        variable_minimum_std={
            str(key): float(value)
            for key, value in dict(variable_minimum_std).items()
        },
    )

    train_dataset = VeinCastERA5Dataset(
        dynamic_path=data_config["dynamic_path"],
        registry=registry,
        start=data_config["train_start"],
        end=data_config["train_end"],
        normalizer=normalizer,
        fit_normalizer=normalizer is None,
        static_path=data_config.get("static_path"),
        static_variables=data_config.get("static_variables", ()),
        surface_pressure_name=data_config.get("surface_pressure_name", "sp"),
        data_interval_hours=data_config["data_interval_hours"],
        base_lead_hours=data_config["base_lead_hours"],
        rollout_steps=data_config["rollout_steps"],
        input_dropout_probability=data_config.get("input_dropout_probability", 0.0),
        training=True,
        load_into_memory=data_config.get("load_into_memory", False),
    )
    normalizer = train_dataset.normalizer
    if not stats_path.exists():
        if not distributed or settings.is_main:
            normalizer.save(stats_path)
        if distributed:
            dist.barrier()

    valid_dataset = VeinCastERA5Dataset(
        dynamic_path=data_config["dynamic_path"],
        registry=registry,
        start=data_config["valid_start"],
        end=data_config["valid_end"],
        normalizer=normalizer,
        static_path=data_config.get("static_path"),
        static_variables=data_config.get("static_variables", ()),
        surface_pressure_name=data_config.get("surface_pressure_name", "sp"),
        data_interval_hours=data_config["data_interval_hours"],
        base_lead_hours=data_config["base_lead_hours"],
        rollout_steps=data_config["rollout_steps"],
        training=False,
        load_into_memory=data_config.get("load_into_memory", False),
    )
    return train_dataset, valid_dataset, normalizer


def run_epoch(
    model: VeinCast,
    loader: DataLoader,
    criterion: VeinCastForecastLoss,
    device: torch.device,
    settings: DistributedSettings | None = None,
    optimizer: torch.optim.Optimizer = None,
    scaler: torch.amp.GradScaler = None,
    teacher_forcing_probability: float = 0.0,
    detach_rollout: bool = False,
    gradient_clip: float = 1.0,
    amp: bool = True,
    amp_dtype: str | None = None,
    epoch: int = 0,
    phase: str = "train",
    show_progress: bool = True,
    log_interval: int = 10,
    activation_checkpointing: bool = False,
    checkpoint_preserve_rng_state: bool = False,
    gradient_accumulation_steps: int = 1,
    diagnostic_field_names: Tuple[str, ...] = (),
    diagnostic_loss_threshold: float = 0.0,
    diagnostic_top_k: int = 10,
    diagnostic_log_interval: int = 0,
    abort_on_loss_spike: bool = True,
    diagnostic_all_ranks_on_spike: bool = False,
    gradient_log_interval: int = 0,
    gradient_log_loss_fraction: float = 0.0,
    skip_update_grad_norm: float = 0.0,
    log_diagnostics: bool = False,
) -> Dict[str, float]:
    training = optimizer is not None
    gradient_accumulation_steps = max(int(gradient_accumulation_steps), 1)
    rank = settings.rank if settings is not None else 0
    model.train(training)
    accumulator = MetricAccumulator()
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    use_tqdm = show_progress and tqdm is not None
    progress = (
        tqdm(
            loader,
            total=total_batches,
            desc=f"{phase} epoch {epoch}",
            dynamic_ncols=True,
            leave=True,
            mininterval=2.0,
        )
        if use_tqdm
        else loader
    )

    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, raw_batch in enumerate(progress, start=1):
        batch = move_batch(raw_batch, device)
        batch_size = int(batch["state"].shape[0])
        current_state = batch["state"]
        current_present = batch["input_present"]
        targets = batch["targets"]
        rollout_steps = targets.shape[1]
        total_loss = current_state.new_zeros(())
        batch_components = MetricAccumulator()
        last_prediction = None
        last_target = None
        last_valid_mask = None

        if total_batches is None:
            accumulation_count = gradient_accumulation_steps
            should_update = batch_index % gradient_accumulation_steps == 0
        else:
            accumulation_start = (
                (batch_index - 1) // gradient_accumulation_steps
            ) * gradient_accumulation_steps + 1
            accumulation_count = min(
                gradient_accumulation_steps,
                total_batches - accumulation_start + 1,
            )
            should_update = (
                batch_index % gradient_accumulation_steps == 0
                or batch_index == total_batches
            )

        sync_context = nullcontext()
        if (
            training
            and not should_update
            and hasattr(model, "no_sync")
        ):
            sync_context = model.no_sync()

        context = torch.enable_grad() if training else torch.no_grad()
        with sync_context:
            with context:
                for step in range(rollout_steps):
                    calendar_features = None
                    if "calendar_features" in batch:
                        calendar_features = batch["calendar_features"][:, step]
                    with autocast_context(device, enabled=amp, amp_dtype=amp_dtype):
                        output = forward_model_step(
                            model=model,
                            state=current_state,
                            static=batch["static"],
                            input_present=current_present,
                            lead_hours=batch["lead_hours"],
                            calendar_features=calendar_features,
                            use_activation_checkpoint=activation_checkpointing,
                            preserve_rng_state=checkpoint_preserve_rng_state,
                        )
                        target = targets[:, step]
                        step_loss, components = criterion(
                            prediction=output["prediction"],
                            target=target,
                            valid_mask=batch["valid_mask"][:, step],
                        )
                        total_loss = total_loss + step_loss / rollout_steps
                        last_prediction = output["prediction"]
                        last_target = target
                        last_valid_mask = batch["valid_mask"][:, step]

                    batch_components.update(components, batch_size)
                    prediction_for_rollout = output["prediction"]
                    if detach_rollout:
                        prediction_for_rollout = prediction_for_rollout.detach()
                    if training and teacher_forcing_probability > 0:
                        teacher = (
                            torch.rand(batch_size, 1, 1, 1, device=device)
                            < teacher_forcing_probability
                        )
                        current_state = torch.where(
                            teacher, target, prediction_for_rollout
                        )
                    else:
                        current_state = prediction_for_rollout
                    current_present = torch.ones_like(current_present)

            pre_components = batch_components.compute()
            pre_components["rollout_total"] = total_loss.detach()
            pre_batch_log = {
                key: metric_to_float(value)
                for key, value in pre_components.items()
                if key in {"total", "forecast_huber", "forecast_mse", "rollout_total"}
            }
            batch_loss = pre_batch_log.get("rollout_total", pre_batch_log.get("total", 0.0))
            threshold = float(diagnostic_loss_threshold)
            loss_spike = (not math.isfinite(batch_loss)) or (
                threshold > 0 and batch_loss > threshold
            )
            global_loss_spike = (
                any_rank_true(loss_spike, device, settings)
                if diagnostic_all_ranks_on_spike
                else loss_spike
            )
            periodic_diagnostic = (
                int(diagnostic_log_interval) > 0
                and batch_index % int(diagnostic_log_interval) == 0
            )
            if (
                (
                    (periodic_diagnostic and log_diagnostics)
                    or (
                        global_loss_spike
                        and (diagnostic_all_ranks_on_spike or log_diagnostics)
                    )
                )
                and last_prediction is not None
                and last_target is not None
                and last_valid_mask is not None
            ):
                print(
                    json.dumps(
                        {
                            "event": "batch_diagnostic",
                            "phase": phase,
                            "rank": rank,
                            "epoch": epoch,
                            "batch": batch_index,
                            "batches": total_batches,
                            "loss_spike": global_loss_spike,
                            "local_loss_spike": loss_spike,
                            "threshold": threshold,
                            "metrics": pre_batch_log,
                            "diagnostic": per_field_diagnostic(
                                last_prediction,
                                last_target,
                                last_valid_mask,
                                diagnostic_field_names,
                                diagnostic_top_k,
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if global_loss_spike and abort_on_loss_spike:
                raise RuntimeError(
                    f"{phase} loss spike at epoch {epoch} batch {batch_index}: "
                    f"local_loss={batch_loss:.6g}, threshold={threshold:.6g}"
                )

            if training:
                loss_for_backward = total_loss / accumulation_count
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

        if training and should_update:
            high_loss_for_gradient_log = (
                threshold > 0
                and float(gradient_log_loss_fraction) > 0
                and math.isfinite(batch_loss)
                and batch_loss >= threshold * float(gradient_log_loss_fraction)
            )
            periodic_gradient_log = (
                int(gradient_log_interval) > 0
                and (batch_index == 1 or batch_index % int(gradient_log_interval) == 0)
            )
            should_log_gradient = (
                (periodic_gradient_log and log_diagnostics)
                or high_loss_for_gradient_log
            )
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
                if gradient_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip
                    )
                else:
                    grad_norm = compute_gradient_norm(model)
            else:
                if gradient_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip
                    )
                else:
                    grad_norm = compute_gradient_norm(model)
            grad_norm_value = metric_tensor_to_float(grad_norm)
            skip_update = (not math.isfinite(grad_norm_value)) or (
                float(skip_update_grad_norm) > 0
                and grad_norm_value > float(skip_update_grad_norm)
            )
            if scaler is not None and scaler.is_enabled():
                if not skip_update:
                    scaler.step(optimizer)
                scaler.update()
            elif not skip_update:
                optimizer.step()
            should_log_gradient = should_log_gradient or skip_update
            if should_log_gradient:
                print(
                    json.dumps(
                        {
                            "event": "gradient_diagnostic",
                            "phase": phase,
                            "rank": rank,
                            "epoch": epoch,
                            "batch": batch_index,
                            "batches": total_batches,
                            "loss": batch_loss,
                            "grad_norm_before_clip": metric_tensor_to_float(grad_norm),
                            "gradient_clip": float(gradient_clip),
                            "skip_update_grad_norm": float(skip_update_grad_norm),
                            "skipped_update": bool(skip_update),
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "high_loss_trigger": high_loss_for_gradient_log,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            optimizer.zero_grad(set_to_none=True)

        components = batch_components.compute()
        components["rollout_total"] = total_loss.detach()
        accumulator.update(
            {key: torch.as_tensor(value) for key, value in components.items()},
            batch_size,
        )
        batch_log = {
            key: metric_to_float(value)
            for key, value in components.items()
            if key in {"total", "forecast_huber", "forecast_mse", "rollout_total"}
        }
        if use_tqdm:
            progress.set_postfix(
                {key: f"{value:.4g}" for key, value in batch_log.items()}
            )
        if show_progress and (batch_index == 1 or batch_index % max(log_interval, 1) == 0):
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches": total_batches,
                        "metrics": batch_log,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return accumulator.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the VeinCast weather forecaster")
    parser.add_argument("--config", default="configs/veincast_stage1.json")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init-from",
        default=None,
        help="Load model weights only and start a fresh optimizer/scheduler state.",
    )
    parser.add_argument(
        "--init-strict",
        dest="init_strict",
        action="store_true",
        default=True,
        help="Require exact checkpoint/model key match when using --init-from.",
    )
    parser.add_argument(
        "--no-init-strict",
        dest="init_strict",
        action="store_false",
        help="Allow missing/unexpected model keys when using --init-from.",
    )
    args = parser.parse_args()
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")

    settings, device = setup_distributed()
    log_memory_stage("after_setup_distributed", device, settings)
    config = resolve_config_paths(load_config(args.config))
    set_seed(int(config["training"]["seed"]) + settings.rank)
    registry = VariableRegistry()
    train_dataset = None
    valid_dataset = None
    try:
        log_memory_stage("before_build_datasets", device, settings)
        train_dataset, valid_dataset, normalizer = build_datasets(
            config, registry, settings
        )
        log_memory_stage("after_build_datasets", device, settings)

        config["model"]["image_size"] = [
            len(train_dataset.latitude), len(train_dataset.longitude)
        ]
        config["model"]["static_channels"] = train_dataset.static_channels
        output_dir = Path(config["training"]["output_dir"])
        if settings.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_config(config, output_dir / "resolved_config.json")
            with (output_dir / "registry.json").open("w", encoding="utf-8") as handle:
                json.dump(registry.to_dict(), handle, ensure_ascii=False, indent=2)
        if settings.enabled:
            dist.barrier()

        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=settings.world_size,
                rank=settings.rank,
                shuffle=True,
                drop_last=True,
            )
            if settings.enabled
            else None
        )
        valid_sampler = (
            DistributedSampler(
                valid_dataset,
                num_replicas=settings.world_size,
                rank=settings.rank,
                shuffle=False,
                drop_last=False,
            )
            if settings.enabled
            else None
        )
        training_config = config["training"]
        torch.set_float32_matmul_precision(
            str(training_config.get("float32_matmul_precision", "high"))
        )
        num_workers = int(training_config["num_workers"])
        loader_kwargs = {
            "batch_size": int(training_config["batch_size"]),
            "num_workers": num_workers,
            "pin_memory": bool(
                training_config.get("pin_memory", device.type == "cuda")
            ),
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(
                training_config.get("persistent_workers", False)
            )
            loader_kwargs["prefetch_factor"] = int(
                training_config.get("prefetch_factor", 2)
            )
        if settings.is_main:
            print(
                json.dumps(
                    {
                        "training_batch_size": {
                            "per_rank_batch_size": loader_kwargs["batch_size"],
                            "world_size": settings.world_size,
                            "global_batch_size": loader_kwargs["batch_size"] * settings.world_size,
                            "mode": "ddp_per_rank",
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        train_loader = DataLoader(
            train_dataset,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            **loader_kwargs,
        )
        valid_loader = DataLoader(
            valid_dataset,
            shuffle=False,
            sampler=valid_sampler,
            drop_last=False,
            **loader_kwargs,
        )
        log_memory_stage("after_build_loaders", device, settings)

        base_model = VeinCast(registry, config["model"]).to(device)
        log_memory_stage("after_model_to_device", device, settings)
        model: torch.nn.Module = base_model
        if settings.enabled:
            ddp_kwargs = (
                {"device_ids": [settings.local_rank], "output_device": settings.local_rank}
                if device.type == "cuda"
                else {}
            )
            ddp_kwargs.update(
                {
                    "broadcast_buffers": bool(
                        training_config.get("ddp_broadcast_buffers", False)
                    ),
                    "gradient_as_bucket_view": bool(
                        training_config.get("ddp_gradient_as_bucket_view", True)
                    ),
                    "static_graph": bool(
                        training_config.get("ddp_static_graph", False)
                    ),
                    "bucket_cap_mb": int(
                        training_config.get("ddp_bucket_cap_mb", 25)
                    ),
                    "find_unused_parameters": False,
                }
            )
            model = DistributedDataParallel(base_model, **ddp_kwargs)
            log_memory_stage("after_ddp_wrap", device, settings)
        criterion = VeinCastForecastLoss(
            train_dataset.latitude,
            config["loss"],
        ).to(device)
        optimizer, optimizer_summary = build_optimizer(model, training_config)
        if settings.is_main:
            amp_dtype = str(training_config.get("amp_dtype", "float16"))
            resolved_amp_dtype = (
                str(resolve_amp_dtype(device, amp_dtype)).replace("torch.", "")
                if bool(training_config["amp"]) and device.type == "cuda"
                else "disabled"
            )
            print(
                json.dumps(
                    {
                        "optimizer": optimizer_summary,
                        "amp": {
                            "enabled": bool(training_config["amp"]),
                            "requested_dtype": amp_dtype,
                            "resolved_dtype": resolved_amp_dtype,
                            "grad_scaler_enabled": (
                                bool(training_config["amp"])
                                and resolved_amp_dtype == "float16"
                            ),
                        },
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        log_memory_stage("after_optimizer", device, settings)
        epochs = int(config["training"]["epochs"])
        warmup = int(config["training"].get("warmup_epochs", 0))

        def learning_rate_multiplier(epoch: int) -> float:
            if warmup and epoch < warmup:
                return float(epoch + 1) / warmup
            progress = (epoch - warmup) / max(epochs - warmup - 1, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, learning_rate_multiplier
        )
        scaler = make_grad_scaler(
            device,
            enabled=bool(config["training"]["amp"]),
            amp_dtype=str(config["training"].get("amp_dtype", "float16")),
        )
        start_epoch = 0
        best_valid = float("inf")
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch, best_valid = restore_training_state(
                checkpoint, unwrap_model(model), optimizer, scheduler, scaler
            )
        elif args.init_from:
            initialize_model_weights(
                args.init_from,
                unwrap_model(model),
                device,
                strict=bool(args.init_strict),
            )
            if settings.is_main:
                print(f"Initialized model weights from {args.init_from}", flush=True)

        history_path = output_dir / "history.json"
        if settings.is_main and args.resume and history_path.exists():
            with history_path.open("r", encoding="utf-8") as handle:
                history = json.load(handle)
        else:
            history = []
        epochs_without_improvement = 0
        for epoch in range(start_epoch, epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            log_memory_stage(f"before_train_epoch_{epoch}", device, settings)
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                settings=settings,
                optimizer=optimizer,
                scaler=scaler,
                teacher_forcing_probability=float(
                    config["training"]["teacher_forcing_probability"]
                ),
                detach_rollout=bool(config["training"]["detach_rollout"]),
                gradient_clip=float(config["training"]["gradient_clip"]),
                amp=bool(config["training"]["amp"]),
                amp_dtype=str(config["training"].get("amp_dtype", "float16")),
                epoch=epoch,
                phase="train",
                show_progress=settings.is_main and bool(config["training"].get("progress", True)),
                log_interval=int(config["training"].get("progress_log_interval", 10)),
                activation_checkpointing=bool(
                    config["training"].get("activation_checkpointing", False)
                ),
                checkpoint_preserve_rng_state=bool(
                    config["training"].get("checkpoint_preserve_rng_state", False)
                ),
                gradient_accumulation_steps=int(
                    config["training"].get("gradient_accumulation_steps", 1)
                ),
                diagnostic_field_names=tuple(field.key for field in registry.fields),
                diagnostic_loss_threshold=float(
                    config["training"].get("loss_spike_threshold", 0.0)
                ),
                diagnostic_top_k=int(config["training"].get("diagnostic_top_k", 10)),
                diagnostic_log_interval=int(
                    config["training"].get("diagnostic_log_interval", 0)
                ),
                abort_on_loss_spike=bool(
                    config["training"].get("abort_on_loss_spike", True)
                ),
                diagnostic_all_ranks_on_spike=bool(
                    config["training"].get("diagnostic_all_ranks_on_spike", True)
                ),
                gradient_log_interval=int(
                    config["training"].get("gradient_log_interval", 0)
                ),
                gradient_log_loss_fraction=float(
                    config["training"].get("gradient_log_loss_fraction", 0.0)
                ),
                skip_update_grad_norm=float(
                    config["training"].get("skip_update_grad_norm", 0.0)
                ),
                log_diagnostics=settings.is_main,
            )
            log_memory_stage(f"before_valid_epoch_{epoch}", device, settings)
            valid_metrics = run_epoch(
                model,
                valid_loader,
                criterion,
                device,
                settings=settings,
                amp=bool(config["training"]["amp"]),
                amp_dtype=str(config["training"].get("amp_dtype", "float16")),
                epoch=epoch,
                phase="valid",
                show_progress=settings.is_main and bool(config["training"].get("progress", True)),
                log_interval=int(config["training"].get("progress_log_interval", 10)),
                activation_checkpointing=False,
                gradient_accumulation_steps=1,
                diagnostic_field_names=tuple(field.key for field in registry.fields),
                diagnostic_loss_threshold=float(
                    config["training"].get("loss_spike_threshold", 0.0)
                ),
                diagnostic_top_k=int(config["training"].get("diagnostic_top_k", 10)),
                diagnostic_log_interval=0,
                abort_on_loss_spike=bool(
                    config["training"].get("abort_on_loss_spike", True)
                ),
                diagnostic_all_ranks_on_spike=bool(
                    config["training"].get("diagnostic_all_ranks_on_spike", True)
                ),
                log_diagnostics=settings.is_main,
            )
            log_memory_stage(f"after_valid_epoch_{epoch}", device, settings)
            if bool(config["training"].get("empty_cache_each_epoch", False)):
                clear_cuda_cache(f"after_empty_cache_epoch_{epoch}", device, settings)
            train_metrics = reduce_metrics(train_metrics, device, settings)
            valid_metrics = reduce_metrics(valid_metrics, device, settings)
            scheduler.step()
            record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "valid": valid_metrics,
            }
            if settings.is_main:
                history.append(record)
                print(json.dumps(record, ensure_ascii=False))
                with history_path.open("w", encoding="utf-8") as handle:
                    json.dump(history, handle, ensure_ascii=False, indent=2)

            valid_value = valid_metrics["rollout_total"]
            improved = valid_value < best_valid
            if improved:
                best_valid = valid_value
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if settings.is_main:
                checkpoint = make_checkpoint(
                    epoch,
                    unwrap_model(model),
                    optimizer,
                    scheduler,
                    scaler,
                    best_valid,
                    config,
                    registry,
                    normalizer,
                )
                torch.save(checkpoint, output_dir / "last.pt")
                if improved:
                    torch.save(checkpoint, output_dir / "best.pt")
                elif epochs_without_improvement >= int(config["training"]["patience"]):
                    print(f"Early stopping after epoch {epoch}")
            if epochs_without_improvement >= int(config["training"]["patience"]):
                break
    finally:
        if train_dataset is not None:
            train_dataset.close()
        if valid_dataset is not None:
            valid_dataset.close()
        cleanup_distributed(settings)


if __name__ == "__main__":
    main()
