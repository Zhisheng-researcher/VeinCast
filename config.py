from __future__ import annotations

import json
import posixpath
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent


DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "dynamic_path": "/mnt/data_cpfs/songjingwei/liyuxuan/weather/data/era5_69vars.nc",
        "static_path": "/mnt/data_cpfs/songjingwei/liyuxuan/weather/data/era5_static.nc",
        "static_variables": ["land_sea_mask", "orography"],
        "surface_pressure_name": "sp",
        "train_start": "1979-01-01",
        "train_end": "2017-12-31",
        "valid_start": "2018-01-01",
        "valid_end": "2018-12-31",
        "test_start": "2020-01-01",
        "test_end": "2020-12-31",
        "base_lead_hours": 6,
        "data_interval_hours": 6,
        "rollout_steps": 1,
        "input_dropout_probability": 0.15,
        "load_into_memory": False,
        "stats_path": "artifacts/normalization_era5_69vars_1979_2017.json",
        "minimum_normalization_std": 0.0,
        "variable_minimum_normalization_std": {"q": 1e-4}
    },
    "model": {
        "image_size": [121, 240],
        "patch_size": [4, 4],
        "embed_dim": 96,
        "depths": [2, 4],
        "num_heads": [4, 8],
        "window_size": [4, 8],
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "drop_path": 0.02,
        "graph_topk_residual": 4,
        "graph_prior_strength": 1.0,
        "pressure_fourier_bands": 8,
        "static_channels": 7,
        "surface_stem_depth": 1,
        "atmosphere_stem_depth": 1,
        "fusion_latents": 4,
        "fusion_dim": 384,
        "fusion_depths": [3, 4, 2],
        "fusion_heads": [12, 24, 12],
        "fusion_window_size": [4, 8],
        "fusion_drop_path": 0.02,
        "fusion_feedback_gate_bias": -3.0,
        "fusion_feedback_gate_min": 0.0,
        "fusion_feedback_gate_max": 0.25,
        "query_fusion_gate_bias": -2.0,
        "query_fusion_gate_min": 0.0,
        "query_fusion_gate_max": 0.5,
        "fusion_centrality_floor": 1e-4,
        "use_calendar_embedding": True,
        "calendar_fourier_bands": 8,
        "geo_fourier_bands": 8,
        "bounded_recovery_affine": True,
        "recovery_affine_scale_radius": 0.25,
        "recovery_affine_bias_radius": 0.25,
        "recovery_output_soft_clamp": 8.0,
        "recovery_zero_init": False,
        "recovery_init_std": 1e-3,
        "residual_prediction": True,
        "prediction_residual_scale": 0.25,
        "prediction_output_soft_clamp": 12.0
    },
    "loss": {
        "forecast_weight": 1.0,
        "forecast_loss": "huber",
        "huber_delta": 2.0
    },
    "training": {
        "output_dir": "artifacts/veincast_stage1",
        "batch_size": 8,
        "num_workers": 8,
        "progress": True,
        "progress_log_interval": 10,
        "epochs": 100,
        "learning_rate": 0.0001,
        "weight_decay": 0.001,
        "warmup_epochs": 2,
        "gradient_clip": 1.0,
        "amp": True,
        "amp_dtype": "bfloat16",
        "activation_checkpointing": True,
        "checkpoint_preserve_rng_state": True,
        "gradient_accumulation_steps": 1,
        "sensitive_lr_multiplier": 0.25,
        "skip_update_grad_norm": 5.0,
        "float32_matmul_precision": "high",
        "pin_memory": True,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "ddp_broadcast_buffers": False,
        "ddp_gradient_as_bucket_view": True,
        "ddp_static_graph": False,
        "ddp_bucket_cap_mb": 25,
        "teacher_forcing_probability": 0.0,
        "detach_rollout": True,
        "loss_spike_threshold": 10.0,
        "abort_on_loss_spike": True,
        "diagnostic_all_ranks_on_spike": True,
        "diagnostic_log_interval": 200,
        "diagnostic_top_k": 10,
        "gradient_log_interval": 200,
        "gradient_log_loss_fraction": 0.25,
        "empty_cache_each_epoch": True,
        "patience": 150,
        "seed": 42
    }
}


def _resolve_config_file(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    project_candidate = PROJECT_ROOT / candidate
    return project_candidate if project_candidate.exists() else candidate


def resolve_path(path: str | Path | None, project_root: str | Path = PROJECT_ROOT) -> str | None:
    if path is None:
        return None
    raw = str(path)
    if not raw or "://" in raw:
        return raw
    if raw.startswith("/"):
        return posixpath.normpath(raw)

    base_raw = str(project_root)
    base_posix = base_raw.replace("\\", "/")
    if base_posix.startswith("/") and not Path(project_root).drive:
        return posixpath.normpath(posixpath.join(base_posix, raw))

    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.as_posix()
    return (Path(project_root) / candidate).resolve().as_posix()


def resolve_config_paths(
    config: Dict[str, Any],
    project_root: str | Path = PROJECT_ROOT,
) -> Dict[str, Any]:
    resolved = deepcopy(config)
    for key in ("dynamic_path", "static_path", "stats_path"):
        if key in resolved.get("data", {}):
            resolved["data"][key] = resolve_path(resolved["data"][key], project_root)
    if "output_dir" in resolved.get("training", {}):
        resolved["training"]["output_dir"] = resolve_path(
            resolved["training"]["output_dir"], project_root
        )
    return resolved


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    with _resolve_config_file(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _deep_update(config, payload)


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
