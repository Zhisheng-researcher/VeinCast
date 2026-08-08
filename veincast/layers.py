from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_probability)
        return x * random_tensor / keep_probability


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ContinuousFourierEmbedding(nn.Module):
    """Continuous scalar embedding used for pressure and lead time."""

    def __init__(self, dim: int, num_bands: int = 8, max_frequency: float = 64.0):
        super().__init__()
        frequencies = torch.logspace(0.0, math.log10(max_frequency), num_bands)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(2 * num_bands + 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, value: torch.Tensor, special_flag: Optional[torch.Tensor] = None) -> torch.Tensor:
        value = value.float()
        if special_flag is None:
            special_flag = torch.zeros_like(value)
        angles = value.unsqueeze(-1) * self.frequencies
        features = torch.cat(
            [
                value.unsqueeze(-1),
                special_flag.float().unsqueeze(-1),
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=-1,
        )
        return self.projection(features)


class FieldMetadataEmbedding(nn.Module):
    def __init__(self, num_variables: int, dim: int, pressure_bands: int = 8):
        super().__init__()
        self.variable_embedding = nn.Embedding(num_variables, dim)
        self.pressure_embedding = ContinuousFourierEmbedding(dim, pressure_bands)
        self.surface_embedding = nn.Embedding(2, dim)

    def forward(
        self,
        variable_ids: torch.Tensor,
        pressure_hpa: torch.Tensor,
        is_surface: torch.Tensor,
    ) -> torch.Tensor:
        pressure_coordinate = torch.where(
            is_surface.bool(),
            torch.zeros_like(pressure_hpa),
            torch.log(torch.clamp(pressure_hpa, min=1.0) / 1000.0),
        )
        return (
            self.variable_embedding(variable_ids.long())
            + self.pressure_embedding(pressure_coordinate, is_surface.float())
            + self.surface_embedding(is_surface.long())
        )


class VariablePatchEmbedding(nn.Module):
    """Shared scalar-field patch projection plus variable/pressure metadata."""

    def __init__(
        self,
        num_variables: int,
        dim: int,
        patch_size: Tuple[int, int],
        pressure_bands: int = 8,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.projection = nn.Conv2d(1, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.metadata = FieldMetadataEmbedding(num_variables, dim, pressure_bands)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        nn.init.trunc_normal_(self.missing_token, std=0.02)

    def forward(
        self,
        fields: torch.Tensor,
        variable_ids: torch.Tensor,
        pressure_hpa: torch.Tensor,
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
        grid_h, grid_w = embedded.shape[-2:]
        embedded = embedded.reshape(batch, num_fields, -1, grid_h, grid_w).permute(0, 1, 3, 4, 2)

        metadata = self.metadata(variable_ids, pressure_hpa, is_surface)
        if metadata.ndim == 2:
            metadata = metadata.unsqueeze(0).expand(batch, -1, -1)
        embedded = embedded + metadata[:, :, None, None, :]
        embedded = embedded + (1.0 - field_present[:, :, None, None, None]) * self.missing_token
        return embedded, (grid_h, grid_w)


class StaticPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, dim: int, patch_size: Tuple[int, int]):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.projection = nn.Conv2d(
            in_channels, dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        height, width = static.shape[-2:]
        patch_h, patch_w = self.patch_size
        pad_h = (patch_h - height % patch_h) % patch_h
        pad_w = (patch_w - width % patch_w) % patch_w
        static = F.pad(static, (0, pad_w, 0, pad_h))
        return self.projection(static).permute(0, 2, 3, 1)


def pad_to_window(
    x: torch.Tensor, window_size: Tuple[int, int]
) -> Tuple[torch.Tensor, int, int]:
    """Pad [B, M, H, W, C], using periodic longitude and replicated latitude."""
    height, width = x.shape[2:4]
    window_h, window_w = window_size
    pad_h = (window_h - height % window_h) % window_h
    pad_w = (window_w - width % window_w) % window_w
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    batch, fields, _, _, channels = x.shape
    image = x.permute(0, 1, 4, 2, 3).reshape(batch * fields, channels, height, width)
    if pad_w:
        image = F.pad(image, (0, pad_w, 0, 0), mode="circular")
    if pad_h:
        image = F.pad(image, (0, 0, 0, pad_h), mode="replicate")
    padded = image.reshape(batch, fields, channels, height + pad_h, width + pad_w)
    return padded.permute(0, 1, 3, 4, 2), pad_h, pad_w


def window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
    """[B,M,H,W,C] -> [B,M,R,T,C]. H and W must be divisible by window size."""
    batch, fields, height, width, channels = x.shape
    window_h, window_w = window_size
    windows_h = height // window_h
    windows_w = width // window_w
    x = x.view(batch, fields, windows_h, window_h, windows_w, window_w, channels)
    x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    return x.view(batch, fields, windows_h * windows_w, window_h * window_w, channels)


def window_reverse(
    windows: torch.Tensor,
    window_size: Tuple[int, int],
    height: int,
    width: int,
) -> torch.Tensor:
    batch, fields, _, _, channels = windows.shape
    window_h, window_w = window_size
    windows_h = height // window_h
    windows_w = width // window_w
    x = windows.view(batch, fields, windows_h, windows_w, window_h, window_w, channels)
    x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    return x.view(batch, fields, height, width, channels)


def build_shift_mask(
    grid_size: Tuple[int, int],
    window_size: Tuple[int, int],
    shift_size: Tuple[int, int],
) -> Optional[torch.Tensor]:
    shift_h, shift_w = shift_size
    if shift_h == 0 and shift_w == 0:
        return None
    height, width = grid_size
    window_h, window_w = window_size
    mask = torch.zeros(1, 1, height, width, 1)
    h_slices = (
        slice(0, -window_h),
        slice(-window_h, -shift_h),
        slice(-shift_h, None),
    )
    w_slices = (
        slice(0, -window_w),
        slice(-window_w, -shift_w),
        slice(-shift_w, None),
    )
    counter = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            mask[:, :, h_slice, w_slice, :] = counter
            counter += 1
    mask_windows = window_partition(mask, window_size)[0, 0, :, :, 0]
    difference = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return difference.masked_fill(difference != 0, -100.0).masked_fill(difference == 0, 0.0)


class EarthRelativePositionBias(nn.Module):
    def __init__(
        self,
        window_size: Tuple[int, int],
        num_heads: int,
        num_latitude_windows: int,
    ):
        super().__init__()
        window_h, window_w = window_size
        table_size = (2 * window_h - 1) * (2 * window_w - 1)
        self.window_size = window_size
        self.num_heads = num_heads
        self.relative_table = nn.Parameter(torch.zeros(table_size, num_heads))
        token_count = window_h * window_w
        self.latitude_bias = nn.Parameter(
            torch.zeros(num_latitude_windows, num_heads, token_count, token_count)
        )

        coordinates = torch.stack(
            torch.meshgrid(torch.arange(window_h), torch.arange(window_w), indexing="ij")
        ).flatten(1)
        relative = coordinates[:, :, None] - coordinates[:, None, :]
        relative[0] += window_h - 1
        relative[1] += window_w - 1
        relative[0] *= 2 * window_w - 1
        self.register_buffer("relative_index", relative.sum(0), persistent=False)
        nn.init.trunc_normal_(self.relative_table, std=0.02)
        nn.init.trunc_normal_(self.latitude_bias, std=0.01)

    def forward(self, num_longitude_windows: int) -> torch.Tensor:
        token_count = self.relative_index.shape[0]
        relative = self.relative_table[self.relative_index.reshape(-1)]
        relative = relative.view(token_count, token_count, self.num_heads).permute(2, 0, 1)
        latitude = self.latitude_bias[:, None].expand(
            -1, num_longitude_windows, -1, -1, -1
        )
        latitude = latitude.reshape(-1, self.num_heads, token_count, token_count)
        return latitude + relative.unsqueeze(0)


class EarthWindowBlock(nn.Module):
    """Shifted local Earth attention with a graph-generated key adapter."""

    def __init__(
        self,
        dim: int,
        grid_size: Tuple[int, int],
        num_heads: int,
        window_size: Tuple[int, int],
        shifted: bool,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.grid_size = tuple(grid_size)
        self.num_heads = num_heads
        self.window_size = tuple(window_size)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.shift_size = (
            self.window_size[0] // 2,
            self.window_size[1] // 2,
        ) if shifted else (0, 0)

        padded_h = math.ceil(grid_size[0] / self.window_size[0]) * self.window_size[0]
        padded_w = math.ceil(grid_size[1] / self.window_size[1]) * self.window_size[1]
        self.padded_grid = (padded_h, padded_w)
        self.num_latitude_windows = padded_h // self.window_size[0]
        self.num_longitude_windows = padded_w // self.window_size[1]

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.adapter_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.position_bias = EarthRelativePositionBias(
            self.window_size, num_heads, self.num_latitude_windows
        )
        shift_mask = build_shift_mask(self.padded_grid, self.window_size, self.shift_size)
        self.register_buffer("shift_mask", shift_mask, persistent=False)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_adapter: torch.Tensor,
        key_gate: torch.Tensor,
    ) -> torch.Tensor:
        original_h, original_w = x.shape[2:4]
        shortcut = x
        normalized = self.norm1(x)

        normalized, _, _ = pad_to_window(normalized, self.window_size)
        key_adapter, _, _ = pad_to_window(key_adapter, self.window_size)
        key_gate, _, _ = pad_to_window(key_gate, self.window_size)

        shift_h, shift_w = self.shift_size
        if shift_h or shift_w:
            normalized = torch.roll(normalized, shifts=(-shift_h, -shift_w), dims=(2, 3))
            key_adapter = torch.roll(key_adapter, shifts=(-shift_h, -shift_w), dims=(2, 3))
            key_gate = torch.roll(key_gate, shifts=(-shift_h, -shift_w), dims=(2, 3))

        x_windows = window_partition(normalized, self.window_size)
        adapter_windows = window_partition(key_adapter, self.window_size)
        gate_windows = window_partition(key_gate, self.window_size)
        batch, fields, regions, tokens, _ = x_windows.shape

        qkv = self.qkv(x_windows).view(
            batch, fields, regions, tokens, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(4, 0, 1, 2, 5, 3, 6)
        query, key, value = qkv[0], qkv[1], qkv[2]

        adapter_key = self.adapter_projection(adapter_windows).view(
            batch, fields, regions, tokens, self.num_heads, self.head_dim
        )
        adapter_key = adapter_key.permute(0, 1, 2, 4, 3, 5)
        gate = gate_windows.permute(0, 1, 2, 4, 3).unsqueeze(-1)
        key = gate * key + (1.0 - gate) * adapter_key

        attention = (query * self.scale) @ key.transpose(-2, -1)
        position_bias = self.position_bias(self.num_longitude_windows)
        attention = attention + position_bias[None, None]
        if self.shift_mask is not None:
            attention = attention + self.shift_mask[None, None, :, None]
        attention = self.attention_dropout(attention.softmax(dim=-1))

        output = attention @ value
        output = output.permute(0, 1, 2, 4, 3, 5).reshape(
            batch, fields, regions, tokens, self.dim
        )
        output = self.output_dropout(self.output_projection(output))
        output = window_reverse(output, self.window_size, *self.padded_grid)

        if shift_h or shift_w:
            output = torch.roll(output, shifts=(shift_h, shift_w), dims=(2, 3))
        output = output[:, :, :original_h, :original_w]

        x = shortcut + self.drop_path(output)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class QuerySpatialBlock(nn.Module):
    """Spatial Earth attention for decoded query fields without an external adapter."""

    def __init__(
        self,
        dim: int,
        grid_size: Tuple[int, int],
        num_heads: int,
        window_size: Tuple[int, int],
        shifted: bool,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
    ):
        super().__init__()
        self.block = EarthWindowBlock(
            dim=dim,
            grid_size=grid_size,
            num_heads=num_heads,
            window_size=window_size,
            shifted=shifted,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.ones(
            *x.shape[:-1], self.block.num_heads, device=x.device, dtype=x.dtype
        )
        return self.block(x, key_adapter=x, key_gate=gate)


class PatchMerging(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, fields, height, width, channels = x.shape
        if height % 2 or width % 2:
            image = x.permute(0, 1, 4, 2, 3).reshape(
                batch * fields, channels, height, width
            )
            image = F.pad(image, (0, width % 2, 0, height % 2))
            height, width = image.shape[-2:]
            x = image.reshape(batch, fields, channels, height, width).permute(0, 1, 3, 4, 2)
        merged = torch.cat(
            [x[:, :, 0::2, 0::2], x[:, :, 1::2, 0::2],
             x[:, :, 0::2, 1::2], x[:, :, 1::2, 1::2]],
            dim=-1,
        )
        return self.reduction(self.norm(merged))


class PatchExpand(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        batch, fields, height, width, channels = x.shape
        x = self.projection(x)
        channels = x.shape[-1]
        image = x.permute(0, 1, 4, 2, 3).reshape(batch * fields, channels, height, width)
        image = F.interpolate(image, size=output_size, mode="bilinear", align_corners=False)
        return image.reshape(batch, fields, channels, *output_size).permute(0, 1, 3, 4, 2)


class VariableCrossAttention(nn.Module):
    """Cross-attend query fields to all encoded fields at each spatial patch."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        spatial_chunk_size: int = 256,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.spatial_chunk_size = int(spatial_chunk_size)
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(
        self,
        encoded: torch.Tensor,
        query_metadata: torch.Tensor,
        field_present: torch.Tensor,
        physical_bias: torch.Tensor,
    ) -> torch.Tensor:
        batch, fields, height, width, channels = encoded.shape
        queries = query_metadata.shape[1]
        spatial = height * width

        present = field_present[:, :, None, None, None]
        context = (encoded * present).sum(dim=1) / present.sum(dim=1).clamp_min(1.0)
        encoded_flat = encoded.permute(0, 2, 3, 1, 4).reshape(batch * spatial, fields, channels)
        context_flat = context.reshape(batch * spatial, channels)
        query_flat = query_metadata[:, None, :, :].expand(batch, spatial, queries, channels)
        query_flat = query_flat.reshape(batch * spatial, queries, channels)
        query_flat = query_flat + context_flat[:, None, :]
        batch_ids = torch.arange(batch, device=encoded.device).repeat_interleave(spatial)

        outputs = []
        for start in range(0, batch * spatial, self.spatial_chunk_size):
            end = min(start + self.spatial_chunk_size, batch * spatial)
            chunk_queries = self.query_projection(self.norm1(query_flat[start:end]))
            chunk_keys = self.key_projection(encoded_flat[start:end])
            chunk_values = self.value_projection(encoded_flat[start:end])
            size = end - start

            q = chunk_queries.view(size, queries, self.num_heads, self.head_dim).transpose(1, 2)
            k = chunk_keys.view(size, fields, self.num_heads, self.head_dim).transpose(1, 2)
            v = chunk_values.view(size, fields, self.num_heads, self.head_dim).transpose(1, 2)
            scores = (q * self.scale) @ k.transpose(-2, -1)

            ids = batch_ids[start:end]
            bias = physical_bias if physical_bias.ndim == 3 else physical_bias.unsqueeze(0)
            if bias.shape[0] == 1:
                bias = bias.expand(batch, -1, -1)
            scores = scores + bias[ids, None]
            source_mask = field_present[ids].bool()
            scores = scores.masked_fill(~source_mask[:, None, None, :], -1e4)

            attention = self.dropout(scores.softmax(dim=-1))
            result = attention @ v
            result = result.transpose(1, 2).reshape(size, queries, channels)
            outputs.append(self.output_projection(result))

        output = torch.cat(outputs, dim=0)
        output = query_flat + self.dropout(output)
        output = output + self.mlp(self.norm2(output))
        return output.view(batch, height, width, queries, channels).permute(0, 3, 1, 2, 4)


class FieldPatchRecovery(nn.Module):
    def __init__(
        self,
        dim: int,
        num_variables: int,
        patch_size: Tuple[int, int],
        image_size: Tuple[int, int],
        bounded_affine: bool = False,
        affine_scale_radius: float = 0.5,
        affine_bias_radius: float = 0.5,
        output_soft_clamp: float = 0.0,
        zero_init: bool = False,
        init_std: float = 0.0,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.image_size = tuple(image_size)
        self.bounded_affine = bool(bounded_affine)
        self.affine_scale_radius = float(affine_scale_radius)
        self.affine_bias_radius = float(affine_bias_radius)
        self.output_soft_clamp = float(output_soft_clamp)
        self.init_std = float(init_std)
        self.projection = nn.Linear(dim, self.patch_size[0] * self.patch_size[1])
        self.variable_affine = nn.Embedding(num_variables, 2)
        with torch.no_grad():
            self.variable_affine.weight[:, 0].fill_(0.0 if self.bounded_affine else 1.0)
            self.variable_affine.weight[:, 1].zero_()
            if zero_init:
                self.projection.weight.zero_()
                self.projection.bias.zero_()
            elif self.init_std > 0:
                nn.init.normal_(self.projection.weight, mean=0.0, std=self.init_std)
                self.projection.bias.zero_()

    def _soft_clamp(self, values: torch.Tensor, limit: float) -> torch.Tensor:
        if limit <= 0:
            return values
        return values.new_tensor(limit) * torch.tanh(values / values.new_tensor(limit))

    def _affine(self, query_variable_ids: torch.Tensor, batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        affine = self.variable_affine(query_variable_ids.long())
        if affine.ndim == 2:
            affine = affine.unsqueeze(0).expand(batch, -1, -1)
        if self.bounded_affine:
            scale = 1.0 + self.affine_scale_radius * torch.tanh(affine[..., 0])
            bias = self.affine_bias_radius * torch.tanh(affine[..., 1])
        else:
            scale = affine[..., 0]
            bias = affine[..., 1]
        return scale[..., None, None], bias[..., None, None]

    def forward(self, x: torch.Tensor, query_variable_ids: torch.Tensor) -> torch.Tensor:
        batch, queries, grid_h, grid_w, _ = x.shape
        patch_h, patch_w = self.patch_size
        values = self.projection(x).view(
            batch, queries, grid_h, grid_w, patch_h, patch_w
        )
        values = values.permute(0, 1, 2, 4, 3, 5).reshape(
            batch, queries, grid_h * patch_h, grid_w * patch_w
        )
        values = values[:, :, : self.image_size[0], : self.image_size[1]]
        values = self._soft_clamp(values, self.output_soft_clamp)
        scale, bias = self._affine(query_variable_ids, batch)
        return values * scale + bias
