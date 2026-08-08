from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from ..config import resolve_config_paths, resolve_path
from ..data import FieldNormalizer, VeinCastERA5Dataset
from ..metrics import FieldMetricAccumulator
from ..model import VeinCast
from ..variables import VariableRegistry


def _resolve_amp_dtype(device: torch.device, amp_dtype: str | None = None) -> torch.dtype:
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


def _autocast_context(device: torch.device, enabled: bool, amp_dtype: str | None = None):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast(
        device_type="cuda",
        dtype=_resolve_amp_dtype(device, amp_dtype),
    )


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _metric_value(value: torch.Tensor) -> Optional[float]:
    result = float(value.item())
    return result if math.isfinite(result) else None


def evaluate_rollout(
    model: torch.nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    normalizer: FieldNormalizer,
    registry: VariableRegistry,
    latitude: Sequence[float],
    device: torch.device,
    base_lead_hours: int,
    amp: bool = True,
    amp_dtype: str | None = None,
) -> List[Dict[str, object]]:
    model.eval()
    accumulators: Optional[Dict[int, FieldMetricAccumulator]] = None
    expected_target_steps: Optional[List[int]] = None
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            if "target_steps" in batch:
                target_step_tensor = batch["target_steps"]
                if target_step_tensor.ndim == 2:
                    if not torch.equal(
                        target_step_tensor,
                        target_step_tensor[0:1].expand_as(target_step_tensor),
                    ):
                        raise ValueError("All samples in a batch must share target_steps")
                    target_steps = [int(step) for step in target_step_tensor[0].tolist()]
                else:
                    target_steps = [int(step) for step in target_step_tensor.tolist()]
            else:
                target_steps = list(range(1, batch["targets"].shape[1] + 1))
            if len(target_steps) != batch["targets"].shape[1]:
                raise ValueError("target_steps length must match the target tensor")
            if accumulators is None:
                expected_target_steps = target_steps
                accumulators = {
                    step: FieldMetricAccumulator(registry.num_fields, latitude)
                    for step in target_steps
                }
            elif target_steps != expected_target_steps:
                raise ValueError("All evaluation batches must use the same target_steps")

            current_state = batch["state"]
            current_present = batch["input_present"]
            target_index = {step: index for index, step in enumerate(target_steps)}
            for step in range(1, max(target_steps) + 1):
                calendar_kwargs = {}
                if "calendar_features" in batch:
                    calendar_kwargs["calendar_features"] = batch[
                        "calendar_features"
                    ][:, step - 1]
                with _autocast_context(device, enabled=amp, amp_dtype=amp_dtype):
                    output = model(
                        current_state,
                        batch["static"],
                        input_present=current_present,
                        lead_hours=batch["lead_hours"],
                        **calendar_kwargs,
                    )
                prediction = output["prediction"]
                if step in target_index:
                    index = target_index[step]
                    target = batch["targets"][:, index]
                    prediction_physical = normalizer.denormalize_tensor(prediction.float())
                    target_physical = normalizer.denormalize_tensor(target.float())
                    accumulators[step].update(
                        prediction_physical,
                        target_physical,
                        batch["valid_mask"][:, index],
                    )
                current_state = prediction
                current_present = torch.ones_like(current_present)

    if accumulators is None:
        raise ValueError("Evaluation loader produced no batches")

    records: List[Dict[str, object]] = []
    for step, accumulator in sorted(accumulators.items()):
        metrics = accumulator.compute()
        for field in registry.fields:
            records.append(
                {
                    "lead_hours": step * int(base_lead_hours),
                    "field": field.key,
                    "variable": field.variable,
                    "pressure_hpa": None if field.is_surface else field.pressure_hpa,
                    "is_surface": field.is_surface,
                    "units": field.units,
                    "rmse": _metric_value(metrics["rmse"][field.field_id]),
                    "weighted_acc": _metric_value(metrics["acc"][field.field_id]),
                    "effective_weight": _metric_value(metrics["weight"][field.field_id]),
                }
            )
    return records


def _write_records(records: List[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (output_dir / "metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _parse_leads(value: Optional[str]) -> Optional[set[int]]:
    if value is None:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recursive VeinCast forecasts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-lead-hours", type=int, default=336)
    parser.add_argument(
        "--report-leads",
        default="24,72,120,168,240,336",
        help="Comma-separated lead hours retained in output; use an empty string for all",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dynamic-path", default=None)
    parser.add_argument("--static-path", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default="artifacts/veincast_evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = resolve_config_paths(checkpoint["config"])
    registry = VariableRegistry.from_dict(checkpoint["registry"])
    normalizer_payload = checkpoint["normalizer"]
    normalizer = FieldNormalizer(
        normalizer_payload["field_keys"],
        normalizer_payload["mean"],
        normalizer_payload["std"],
    )
    normalizer.validate(registry)

    data_config = config["data"]
    base_lead = int(data_config["base_lead_hours"])
    if args.max_lead_hours <= 0 or args.max_lead_hours % base_lead != 0:
        raise ValueError(
            f"max-lead-hours must be a positive multiple of {base_lead} hours"
        )
    report_leads = _parse_leads(args.report_leads)
    if report_leads:
        invalid_leads = sorted(
            lead
            for lead in report_leads
            if lead <= 0 or lead > args.max_lead_hours or lead % base_lead != 0
        )
        if invalid_leads:
            raise ValueError(
                f"report-leads must be positive {base_lead}-hour multiples no larger "
                f"than max-lead-hours; invalid values: {invalid_leads}"
            )
        target_steps = sorted(lead // base_lead for lead in report_leads)
    else:
        target_steps = list(range(1, args.max_lead_hours // base_lead + 1))

    dataset = VeinCastERA5Dataset(
        dynamic_path=resolve_path(args.dynamic_path) if args.dynamic_path else data_config["dynamic_path"],
        registry=registry,
        start=args.start or data_config["test_start"],
        end=args.end or data_config["test_end"],
        normalizer=normalizer,
        static_path=resolve_path(args.static_path) if args.static_path else data_config.get("static_path"),
        static_variables=data_config.get("static_variables", ()),
        surface_pressure_name=data_config.get("surface_pressure_name", "sp"),
        data_interval_hours=int(data_config["data_interval_hours"]),
        base_lead_hours=base_lead,
        rollout_steps=args.max_lead_hours // base_lead,
        target_steps=target_steps,
        training=False,
        load_into_memory=bool(data_config.get("load_into_memory", False)),
    )
    try:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model = VeinCast(registry, config["model"]).to(device)
        model.load_state_dict(checkpoint["model"])
        records = evaluate_rollout(
            model,
            loader,
            normalizer,
            registry,
            dataset.latitude,
            device,
            base_lead,
            amp=bool(config["training"].get("amp", True)),
            amp_dtype=str(config["training"].get("amp_dtype", "float16")),
        )
        if not records:
            raise ValueError("No metric records match report-leads")
        output_dir = Path(args.output_dir)
        _write_records(records, output_dir)
        print(f"Saved {len(records)} field/lead metric rows to {output_dir}")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
