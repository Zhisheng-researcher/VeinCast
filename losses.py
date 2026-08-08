from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn


def latitude_weights(latitude: torch.Tensor) -> torch.Tensor:
    """Cosine-latitude weights normalized to unit mean."""

    weights = torch.cos(torch.deg2rad(latitude.float())).clamp_min(0.0)
    return weights / weights.mean().clamp_min(1e-6)


class VeinCastForecastLoss(nn.Module):
    """Masked latitude-weighted Huber objective used for reported results."""

    def __init__(
        self,
        latitude: Sequence[float],
        loss_config: Dict[str, float],
    ) -> None:
        super().__init__()
        self.config = dict(loss_config)
        self.register_buffer(
            "lat_weights",
            latitude_weights(torch.as_tensor(latitude)).view(1, 1, -1, 1),
            persistent=True,
        )

    def _weighted_huber(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        error = prediction.float() - target.float()
        delta = max(float(self.config.get("huber_delta", 2.0)), 1e-6)
        absolute = error.abs()
        quadratic = torch.minimum(absolute, absolute.new_tensor(delta))
        linear = absolute - quadratic
        huber = 0.5 * quadratic.square() + delta * linear
        mask = valid_mask.float() * self.lat_weights.float()
        return (huber * mask).sum() / mask.sum().clamp_min(1.0)

    def _weighted_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = valid_mask.float() * self.lat_weights.float()
        squared = (prediction.float() - target.float()).square() * mask
        return squared.sum() / mask.sum().clamp_min(1.0)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        forecast = self._weighted_huber(prediction, target, valid_mask)
        total = float(self.config.get("forecast_weight", 1.0)) * forecast
        components = {
            "total": total.detach(),
            "forecast_huber": forecast.detach(),
            # Retained for monitoring only; it is not part of the objective.
            "forecast_mse": self._weighted_mse(
                prediction, target, valid_mask
            ).detach(),
        }
        return total, components
