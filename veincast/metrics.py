from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch


def _latitude_weights(latitude: torch.Tensor) -> torch.Tensor:
    weights = torch.cos(torch.deg2rad(latitude.float())).clamp_min(0.0)
    return weights / weights.mean().clamp_min(1e-6)


def weighted_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitude: Sequence[float],
) -> torch.Tensor:
    latitude = torch.as_tensor(latitude, device=prediction.device)
    weights = _latitude_weights(latitude).view(1, 1, -1, 1)
    return ((prediction - target).square() * weights).mean(dim=(0, 2, 3)).sqrt()


def weighted_acc(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitude: Sequence[float],
) -> torch.Tensor:
    latitude = torch.as_tensor(latitude, device=prediction.device)
    weights = _latitude_weights(latitude).view(1, 1, -1, 1)
    prediction_anomaly = prediction - prediction.mean(dim=(0, 2, 3), keepdim=True)
    target_anomaly = target - target.mean(dim=(0, 2, 3), keepdim=True)
    covariance = (prediction_anomaly * target_anomaly * weights).mean(dim=(0, 2, 3))
    prediction_scale = (
        prediction_anomaly.square() * weights
    ).mean(dim=(0, 2, 3)).sqrt()
    target_scale = (target_anomaly.square() * weights).mean(dim=(0, 2, 3)).sqrt()
    return covariance / (prediction_scale * target_scale).clamp_min(1e-8)


class MetricAccumulator:
    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.count = 0

    def update(self, values: Dict[str, torch.Tensor], batch_size: int) -> None:
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value.mean().item()) * batch_size
        self.count += batch_size

    def compute(self) -> Dict[str, float]:
        denominator = max(self.count, 1)
        return {key: value / denominator for key, value in self.sums.items()}


class FieldMetricAccumulator:
    """Online latitude-weighted RMSE and correlation for each forecast field."""

    def __init__(self, num_fields: int, latitude: Sequence[float]):
        self.num_fields = int(num_fields)
        self.latitude = torch.as_tensor(latitude, dtype=torch.float64)
        self.statistics = {
            name: torch.zeros(self.num_fields, dtype=torch.float64)
            for name in ("weight", "squared_error", "prediction", "target", "prediction2", "target2", "cross")
        }

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if prediction.shape != target.shape or prediction.ndim != 4:
            raise ValueError("prediction and target must have shape [batch, field, lat, lon]")
        if prediction.shape[1] != self.num_fields:
            raise ValueError(
                f"Expected {self.num_fields} fields, got {prediction.shape[1]}"
            )
        if prediction.shape[2] != len(self.latitude):
            raise ValueError("Latitude coordinate does not match prediction height")
        if valid_mask is None:
            valid_mask = torch.ones_like(prediction)
        elif valid_mask.shape != prediction.shape:
            raise ValueError("valid_mask must match prediction shape")

        latitude = self.latitude.to(device=prediction.device, dtype=prediction.dtype)
        weights = torch.cos(torch.deg2rad(latitude)).clamp_min(0.0).view(1, 1, -1, 1)
        weights = weights * valid_mask.to(dtype=prediction.dtype)
        dimensions = (0, 2, 3)
        batch_statistics = {
            "weight": weights.sum(dim=dimensions),
            "squared_error": ((prediction - target).square() * weights).sum(dim=dimensions),
            "prediction": (prediction * weights).sum(dim=dimensions),
            "target": (target * weights).sum(dim=dimensions),
            "prediction2": (prediction.square() * weights).sum(dim=dimensions),
            "target2": (target.square() * weights).sum(dim=dimensions),
            "cross": (prediction * target * weights).sum(dim=dimensions),
        }
        for name, value in batch_statistics.items():
            self.statistics[name] += value.detach().double().cpu()

    def compute(self) -> Dict[str, torch.Tensor]:
        weight = self.statistics["weight"]
        safe_weight = weight.clamp_min(1e-12)
        rmse = (self.statistics["squared_error"] / safe_weight).sqrt()

        covariance = (
            self.statistics["cross"]
            - self.statistics["prediction"] * self.statistics["target"] / safe_weight
        )
        prediction_variance = (
            self.statistics["prediction2"]
            - self.statistics["prediction"].square() / safe_weight
        ).clamp_min(0.0)
        target_variance = (
            self.statistics["target2"]
            - self.statistics["target"].square() / safe_weight
        ).clamp_min(0.0)
        scale = (prediction_variance * target_variance).sqrt()
        acc = torch.where(
            scale > 1e-12,
            covariance / scale.clamp_min(1e-12),
            torch.full_like(scale, float("nan")),
        )
        return {"rmse": rmse, "acc": acc, "weight": weight}
