from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import FieldMetadataEmbedding
from .variables import VariableRegistry


class LightweightRefinementBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, groups=dim, bias=False)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, 1, 1), mode="replicate")
        x = self.depthwise(x)
        x = self.pointwise(self.activation(x))
        return residual + x


class LightweightFieldStem(nn.Module):
    def __init__(
        self,
        num_variables: int,
        dim: int,
        patch_size: Tuple[int, int],
        pressure_bands: int,
        refinement_depth: int,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.projection = nn.Conv2d(
            1, dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.metadata = FieldMetadataEmbedding(num_variables, dim, pressure_bands)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.refinement = nn.ModuleList(
            LightweightRefinementBlock(dim) for _ in range(refinement_depth)
        )
        self.output_norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.missing_token, std=0.02)

    def forward(
        self,
        fields: torch.Tensor,
        variable_ids: torch.Tensor,
        pressures: torch.Tensor,
        is_surface: torch.Tensor,
        field_present: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        batch, num_fields, height, width = fields.shape
        patch_h, patch_w = self.patch_size
        pad_h = (patch_h - height % patch_h) % patch_h
        pad_w = (patch_w - width % patch_w) % patch_w
        flattened = fields.reshape(batch * num_fields, 1, height, width)
        flattened = F.pad(flattened, (0, pad_w, 0, pad_h))
        embedded = self.projection(flattened)
        for block in self.refinement:
            embedded = block(embedded)
        grid_h, grid_w = embedded.shape[-2:]
        embedded = embedded.reshape(
            batch, num_fields, -1, grid_h, grid_w
        ).permute(0, 1, 3, 4, 2)

        metadata = self.metadata(variable_ids, pressures, is_surface)
        if metadata.ndim == 2:
            metadata = metadata.unsqueeze(0).expand(batch, -1, -1)
        embedded = embedded + metadata[:, :, None, None]
        embedded = embedded + (
            1.0 - field_present[:, :, None, None, None]
        ) * self.missing_token
        return self.output_norm(embedded), (grid_h, grid_w)


class DualModalityStem(nn.Module):
    def __init__(
        self,
        registry: VariableRegistry,
        dim: int,
        patch_size: Tuple[int, int],
        pressure_bands: int = 8,
        refinement_depth: int = 1,
        surface_depth: int | None = None,
        atmosphere_depth: int | None = None,
    ):
        super().__init__()
        surface_indices = [field.field_id for field in registry.fields if field.is_surface]
        atmosphere_indices = [
            field.field_id for field in registry.fields if not field.is_surface
        ]
        if set(surface_indices) & set(atmosphere_indices):
            raise ValueError("Surface and atmosphere field partitions overlap")
        if set(surface_indices) | set(atmosphere_indices) != set(
            range(registry.num_fields)
        ):
            raise ValueError("Stem field partitions do not cover the registry")

        self.num_fields = registry.num_fields
        self.register_buffer(
            "surface_indices", torch.tensor(surface_indices, dtype=torch.long)
        )
        self.register_buffer(
            "atmosphere_indices", torch.tensor(atmosphere_indices, dtype=torch.long)
        )
        stem_args = dict(
            num_variables=registry.num_variables,
            dim=dim,
            patch_size=patch_size,
            pressure_bands=pressure_bands,
        )
        self.surface_stem = LightweightFieldStem(
            **stem_args,
            refinement_depth=(
                refinement_depth if surface_depth is None else int(surface_depth)
            ),
        )
        self.atmosphere_stem = LightweightFieldStem(
            **stem_args,
            refinement_depth=(
                refinement_depth
                if atmosphere_depth is None
                else int(atmosphere_depth)
            ),
        )

    def merge(
        self, surface_features: torch.Tensor, atmosphere_features: torch.Tensor
    ) -> torch.Tensor:
        batch, _, height, width, dim = surface_features.shape
        merged = surface_features.new_zeros(
            batch, self.num_fields, height, width, dim
        )
        merged = merged.index_copy(1, self.surface_indices, surface_features)
        return merged.index_copy(1, self.atmosphere_indices, atmosphere_features)

    def forward(
        self,
        state: torch.Tensor,
        variable_ids: torch.Tensor,
        pressures: torch.Tensor,
        is_surface: torch.Tensor,
        field_present: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        surface_features, surface_grid = self.surface_stem(
            state[:, self.surface_indices],
            variable_ids[..., self.surface_indices],
            pressures[..., self.surface_indices],
            is_surface[..., self.surface_indices],
            field_present[:, self.surface_indices],
        )
        atmosphere_features, atmosphere_grid = self.atmosphere_stem(
            state[:, self.atmosphere_indices],
            variable_ids[..., self.atmosphere_indices],
            pressures[..., self.atmosphere_indices],
            is_surface[..., self.atmosphere_indices],
            field_present[:, self.atmosphere_indices],
        )
        if surface_grid != atmosphere_grid:
            raise ValueError(
                f"Stem grids differ: surface={surface_grid}, atmosphere={atmosphere_grid}"
            )
        return self.merge(surface_features, atmosphere_features), surface_grid
