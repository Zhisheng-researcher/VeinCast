from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import resolve_config_paths, resolve_path
from data import FieldNormalizer, VeinCastERA5Dataset
from variables import VariableRegistry
from veincast import VeinCast


def _normalizer_from_checkpoint(checkpoint: dict) -> FieldNormalizer:
    payload = checkpoint["normalizer"]
    return FieldNormalizer(payload["field_keys"], payload["mean"], payload["std"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a closed-set recursive VeinCast forecast for all 69 fields"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lead-hours", type=int, default=24)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--dynamic-path", default=None)
    parser.add_argument("--static-path", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="artifacts/veincast_prediction.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = resolve_config_paths(checkpoint["config"])
    registry = VariableRegistry.from_dict(checkpoint["registry"])
    normalizer = _normalizer_from_checkpoint(checkpoint)
    normalizer.validate(registry)
    if registry.num_fields != 69:
        raise ValueError(
            f"The reported VeinCast configuration requires 69 fields, got {registry.num_fields}"
        )

    data_config = config["data"]
    base_lead = int(data_config["base_lead_hours"])
    if args.lead_hours <= 0 or args.lead_hours % base_lead != 0:
        raise ValueError(
            f"lead-hours must be a positive multiple of the trained {base_lead}-hour step"
        )
    forecast_steps = args.lead_hours // base_lead
    dataset = VeinCastERA5Dataset(
        dynamic_path=(
            resolve_path(args.dynamic_path)
            if args.dynamic_path
            else data_config["dynamic_path"]
        ),
        registry=registry,
        start=args.start or data_config["test_start"],
        end=args.end or data_config["test_end"],
        normalizer=normalizer,
        static_path=(
            resolve_path(args.static_path)
            if args.static_path
            else data_config.get("static_path")
        ),
        static_variables=data_config.get("static_variables", ()),
        surface_pressure_name=data_config.get("surface_pressure_name", "sp"),
        data_interval_hours=int(data_config["data_interval_hours"]),
        base_lead_hours=base_lead,
        rollout_steps=forecast_steps,
        target_steps=[forecast_steps],
        training=False,
        load_into_memory=bool(data_config.get("load_into_memory", False)),
    )
    try:
        if not 0 <= args.sample_index < len(dataset):
            raise IndexError(
                f"sample-index {args.sample_index} is outside [0, {len(dataset) - 1}]"
            )

        model = VeinCast(registry, config["model"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        sample = dataset[args.sample_index]
        state = sample["state"].unsqueeze(0).to(device)
        static = sample["static"].unsqueeze(0).to(device)
        present = sample["input_present"].unsqueeze(0).to(device)
        calendar_features = sample["calendar_features"].to(device)
        step_lead = torch.tensor([float(base_lead)], device=device)
        predictions = []

        with torch.inference_mode():
            for step in range(forecast_steps):
                output = model(
                    state,
                    static,
                    input_present=present,
                    lead_hours=step_lead,
                    calendar_features=calendar_features[step].unsqueeze(0),
                )
                state = output["prediction"]
                present = torch.ones_like(present)
                physical = normalizer.denormalize_tensor(state.float())
                predictions.append(physical[0].cpu().numpy())

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            prediction=np.stack(predictions, axis=0),
            field_labels=np.asarray([field.key for field in registry.fields]),
            latitude=dataset.latitude,
            longitude=dataset.longitude,
            lead_hours=np.arange(1, forecast_steps + 1, dtype=np.int64) * base_lead,
            sample_index=np.asarray(args.sample_index),
        )
        print(
            f"Saved {forecast_steps} VeinCast steps × {registry.num_fields} fields "
            f"to {output_path}"
        )
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
