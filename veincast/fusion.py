from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .dynamic_graph import DynamicGraphState
from .layers import (
    ContinuousFourierEmbedding,
    DropPath,
    MLP,
    PatchExpand,
    PatchMerging,
    QuerySpatialBlock,
)


class GeographicFourierEmbedding(nn.Module):
    def __init__(self, dim: int, grid_size: Tuple[int, int], num_bands: int = 8):
        super().__init__()
        self.grid_size = tuple(grid_size)
        height, width = self.grid_size
        latitude = torch.linspace(-1.0, 1.0, height)
        longitude = torch.arange(width, dtype=torch.float32) / max(width, 1)
        longitude = longitude * 2.0 - 1.0
        lat_grid, lon_grid = torch.meshgrid(latitude, longitude, indexing="ij")
        frequencies = 2.0 ** torch.arange(num_bands, dtype=torch.float32)
        lat_angles = math.pi * lat_grid[..., None] * frequencies
        lon_angles = math.pi * lon_grid[..., None] * frequencies
        features = torch.cat(
            [
                lat_grid[..., None],
                lon_grid[..., None],
                torch.sin(lat_angles),
                torch.cos(lat_angles),
                torch.sin(lon_angles),
                torch.cos(lon_angles),
            ],
            dim=-1,
        )
        self.register_buffer("features", features, persistent=True)
        feature_dim = 2 + 4 * num_bands
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self) -> torch.Tensor:
        return self.projection(self.features)


class CyclicCalendarEmbedding(nn.Module):
    def __init__(self, dim: int, num_bands: int = 8):
        super().__init__()
        self.num_bands = int(num_bands)
        self.unknown_time = nn.Parameter(torch.zeros(dim))
        frequencies = torch.arange(1, self.num_bands + 1, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(4 * self.num_bands, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        nn.init.trunc_normal_(self.unknown_time, std=0.02)

    def forward(
        self,
        calendar_features: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        if calendar_features is None:
            return self.unknown_time.unsqueeze(0).expand(batch_size, -1)
        if calendar_features.ndim != 2 or calendar_features.shape != (batch_size, 2):
            raise ValueError(
                "calendar_features must have shape [B, 2] containing month and UTC hour"
            )

        calendar = calendar_features.float()
        month = calendar[:, 0]
        hour = calendar[:, 1]
        valid = (
            torch.isfinite(calendar).all(dim=-1)
            & month.ge(1.0)
            & month.le(12.0)
            & hour.ge(0.0)
            & hour.le(23.0)
        )
        safe_month = torch.where(valid, month, torch.ones_like(month))
        safe_hour = torch.where(valid, hour, torch.zeros_like(hour))
        month_angle = (
            2.0
            * math.pi
            * (safe_month - 1.0).unsqueeze(-1)
            / 12.0
            * self.frequencies
        )
        hour_angle = (
            2.0
            * math.pi
            * safe_hour.unsqueeze(-1)
            / 24.0
            * self.frequencies
        )
        encoded = self.projection(
            torch.cat(
                [
                    torch.sin(month_angle),
                    torch.cos(month_angle),
                    torch.sin(hour_angle),
                    torch.cos(hour_angle),
                ],
                dim=-1,
            )
        )
        unknown = self.unknown_time.unsqueeze(0).expand(batch_size, -1)
        return torch.where(valid[:, None], encoded, unknown)


class FusionLatentInitializer(nn.Module):
    def __init__(
        self,
        fusion_dim: int,
        num_latents: int,
        grid_size: Tuple[int, int],
        static_dim: int,
        lead_bands: int = 8,
        calendar_bands: int = 8,
        geo_bands: int = 8,
        use_calendar_embedding: bool = True,
    ):
        super().__init__()
        self.fusion_dim = int(fusion_dim)
        self.num_latents = int(num_latents)
        self.grid_size = tuple(grid_size)
        self.use_calendar_embedding = bool(use_calendar_embedding)
        self.latents = nn.Parameter(
            torch.zeros(1, self.num_latents, 1, 1, self.fusion_dim)
        )
        self.static_projection = nn.Sequential(
            nn.LayerNorm(static_dim),
            nn.Linear(static_dim, self.fusion_dim),
        )
        self.geographic_embedding = GeographicFourierEmbedding(
            self.fusion_dim, self.grid_size, geo_bands
        )
        self.lead_embedding = ContinuousFourierEmbedding(
            self.fusion_dim, lead_bands
        )
        self.calendar_embedding = CyclicCalendarEmbedding(
            self.fusion_dim, calendar_bands
        )
        nn.init.trunc_normal_(self.latents, std=0.02)

    def forward(
        self,
        static_tokens: torch.Tensor,
        lead_hours: torch.Tensor,
        calendar_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, height, width, _ = static_tokens.shape
        if (height, width) != self.grid_size:
            raise ValueError(
                f"Expected static grid {self.grid_size}, got {(height, width)}"
            )
        if lead_hours.numel() != batch:
            raise ValueError("lead_hours must contain one value per batch item")

        static = self.static_projection(static_tokens)[:, None]
        geographic = self.geographic_embedding()[None, None]
        lead = self.lead_embedding(lead_hours.float().reshape(batch) / 24.0)
        lead = lead[:, None, None, None]
        if self.use_calendar_embedding:
            calendar = self.calendar_embedding(calendar_features, batch)
        else:
            calendar = static_tokens.new_zeros(batch, self.fusion_dim)
        calendar = calendar[:, None, None, None]
        return self.latents + static + geographic + lead + calendar


class GraphConditionedFieldToLatentAttention(nn.Module):
    def __init__(
        self,
        field_dim: int,
        fusion_dim: int,
        num_heads: int,
        num_latents: int,
        window_size: Tuple[int, int],
        dropout: float = 0.0,
        centrality_floor: float = 1e-8,
    ):
        super().__init__()
        if fusion_dim % num_heads != 0:
            raise ValueError("fusion_dim must be divisible by num_heads")
        self.fusion_dim = int(fusion_dim)
        self.num_heads = int(num_heads)
        self.num_latents = int(num_latents)
        self.head_dim = self.fusion_dim // self.num_heads
        self.window_size = tuple(window_size)
        self.centrality_floor = float(centrality_floor)
        self.latent_norm = nn.LayerNorm(fusion_dim)
        self.field_norm = nn.LayerNorm(2 * field_dim)
        self.field_projection = nn.Linear(2 * field_dim, fusion_dim)
        self.query = nn.Linear(fusion_dim, fusion_dim, bias=False)
        self.key = nn.Linear(fusion_dim, fusion_dim, bias=False)
        self.value = nn.Linear(fusion_dim, fusion_dim, bias=False)
        self.centrality_bias = nn.Sequential(
            nn.Linear(1, max(8, num_latents)),
            nn.SiLU(),
            nn.Linear(max(8, num_latents), num_latents),
        )
        self.output = nn.Linear(fusion_dim, fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def _centrality_map(
        self,
        graph_state: DynamicGraphState,
        height: int,
        width: int,
    ) -> torch.Tensor:
        centrality = graph_state.adjacency.mean(dim=-2)
        batch, regions, fields = centrality.shape
        window_h, window_w = self.window_size
        regions_h = math.ceil(height / window_h)
        regions_w = math.ceil(width / window_w)
        if regions_h * regions_w != regions:
            raise ValueError(
                f"Graph has {regions} regions, expected {regions_h * regions_w}"
            )
        centrality = centrality.reshape(
            batch, regions_h, regions_w, fields
        ).permute(0, 3, 1, 2)
        centrality = centrality.repeat_interleave(window_h, dim=2)
        centrality = centrality.repeat_interleave(window_w, dim=3)
        return centrality[:, :, :height, :width]

    def forward(
        self,
        latents: torch.Tensor,
        field_tokens: torch.Tensor,
        graph_state: DynamicGraphState,
        field_present: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, fields, height, width, _ = field_tokens.shape
        if latents.shape[:4] != (batch, self.num_latents, height, width):
            raise ValueError("Fusion latent and field grids are incompatible")
        if field_present.shape != (batch, fields):
            raise ValueError("field_present must have shape [B, M]")

        enriched = torch.cat([field_tokens, graph_state.context_map], dim=-1)
        enriched = self.field_projection(self.field_norm(enriched))
        queries = self.query(self.latent_norm(latents))
        keys = self.key(enriched)
        values = self.value(enriched)

        queries = queries.reshape(
            batch, self.num_latents, height, width, self.num_heads, self.head_dim
        ).permute(0, 2, 3, 4, 1, 5)
        keys = keys.reshape(
            batch, fields, height, width, self.num_heads, self.head_dim
        ).permute(0, 2, 3, 4, 1, 5)
        values = values.reshape(
            batch, fields, height, width, self.num_heads, self.head_dim
        ).permute(0, 2, 3, 4, 1, 5)
        logits = torch.einsum("bhwqld,bhwqmd->bhwqlm", queries, keys)
        logits = logits * (self.head_dim ** -0.5)

        centrality = self._centrality_map(graph_state, height, width)
        centrality = torch.log(centrality.clamp_min(self.centrality_floor))[..., None]
        graph_bias = self.centrality_bias(centrality)
        graph_bias = graph_bias.permute(0, 2, 3, 4, 1).unsqueeze(3)
        logits = logits + graph_bias
        source_mask = field_present[:, None, None, None, None, :].bool()
        logits = logits.masked_fill(~source_mask, -1e4)
        attention = logits.softmax(dim=-1)
        attention = self.dropout(attention)

        mixed = torch.einsum("bhwqlm,bhwqmd->bhwqld", attention, values)
        mixed = mixed.permute(0, 4, 1, 2, 3, 5).reshape(
            batch, self.num_latents, height, width, self.fusion_dim
        )
        output = latents + self.output(mixed)
        entropy = -(
            attention.clamp_min(1e-8).log() * attention
        ).sum(dim=-1).mean()
        return output, {"fusion_attention_entropy": entropy}


# Backward-compatible import name used before the paper terminology was fixed.
FieldToFusionCrossAttention = GraphConditionedFieldToLatentAttention


class LatentMixingBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Latent mixing dim must be divisible by num_heads")
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, latents, height, width, dim = x.shape
        tokens = x.permute(0, 2, 3, 1, 4).reshape(
            batch * height * width, latents, dim
        )
        normalized = self.norm1(tokens)
        mixed, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + self.drop_path(mixed)
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))
        return tokens.reshape(batch, height, width, latents, dim).permute(
            0, 3, 1, 2, 4
        )


class FusionEarthBlock(nn.Module):
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
        self.spatial = QuerySpatialBlock(
            dim=dim,
            grid_size=grid_size,
            num_heads=num_heads,
            window_size=window_size,
            shifted=shifted,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
        )
        self.latent = LatentMixingBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.latent(self.spatial(x))


class LatentUFusionBackbone(nn.Module):
    def __init__(
        self,
        dim: int,
        grid_size: Tuple[int, int],
        num_latents: int,
        depths: Tuple[int, int, int],
        num_heads: Tuple[int, int, int],
        window_size: Tuple[int, int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.2,
    ):
        super().__init__()
        if len(depths) != 3 or len(num_heads) != 3:
            raise ValueError("Fusion depths and num_heads must contain three stages")
        if num_latents < 1:
            raise ValueError("num_latents must be positive")
        self.dim = int(dim)
        self.grid_size = tuple(grid_size)
        self.num_latents = int(num_latents)
        middle_grid = (
            math.ceil(self.grid_size[0] / 2),
            math.ceil(self.grid_size[1] / 2),
        )
        total_depth = sum(depths)
        rates = torch.linspace(0.0, drop_path, total_depth).tolist()
        rate_index = 0

        def make_stage(
            stage_dim: int,
            stage_grid: Tuple[int, int],
            depth: int,
            heads: int,
        ) -> nn.ModuleList:
            nonlocal rate_index
            blocks = nn.ModuleList()
            for index in range(depth):
                blocks.append(
                    FusionEarthBlock(
                        dim=stage_dim,
                        grid_size=stage_grid,
                        num_heads=heads,
                        window_size=window_size,
                        shifted=index % 2 == 1,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        drop_path=rates[rate_index],
                    )
                )
                rate_index += 1
            return blocks

        self.encoder_blocks = make_stage(
            self.dim, self.grid_size, depths[0], num_heads[0]
        )
        self.downsample = PatchMerging(self.dim)
        self.bottleneck_blocks = make_stage(
            2 * self.dim, middle_grid, depths[1], num_heads[1]
        )
        self.upsample = PatchExpand(2 * self.dim, self.dim)
        self.skip_fusion = nn.Sequential(
            nn.LayerNorm(2 * self.dim),
            nn.Linear(2 * self.dim, self.dim),
            nn.GELU(),
        )
        self.decoder_blocks = make_stage(
            self.dim, self.grid_size, depths[2], num_heads[2]
        )
        self.output_norm = nn.LayerNorm(self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.num_latents or x.shape[2:4] != self.grid_size:
            raise ValueError(
                f"Expected fusion shape [B,{self.num_latents},{self.grid_size[0]},"
                f"{self.grid_size[1]},C], got {tuple(x.shape)}"
            )
        skip = x
        for block in self.encoder_blocks:
            skip = block(skip)
        x = self.downsample(skip)
        for block in self.bottleneck_blocks:
            x = block(x)
        x = self.upsample(x, self.grid_size)
        x = self.skip_fusion(torch.cat([x, skip], dim=-1))
        for block in self.decoder_blocks:
            x = block(x)
        return self.output_norm(x)


class FusionToFieldFeedback(nn.Module):
    def __init__(
        self,
        field_dim: int,
        fusion_dim: int,
        num_heads: int,
        gate_bias: float = -2.0,
        dropout: float = 0.0,
        gate_min: float = 0.0,
        gate_max: float = 1.0,
    ):
        super().__init__()
        if field_dim % num_heads != 0:
            raise ValueError("field_dim must be divisible by num_heads")
        self.field_dim = int(field_dim)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.field_norm = nn.LayerNorm(field_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=field_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=fusion_dim,
            vdim=fusion_dim,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * field_dim),
            nn.Linear(3 * field_dim, field_dim),
            nn.SiLU(),
            nn.Linear(field_dim, 1),
        )
        nn.init.normal_(self.gate[-1].weight, std=0.01)
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def _bounded_gate(self, logits: torch.Tensor) -> torch.Tensor:
        if self.gate_max < self.gate_min:
            raise ValueError("gate_max must be greater than or equal to gate_min")
        span = self.gate_max - self.gate_min
        return self.gate_min + span * logits.sigmoid()

    def forward(
        self,
        field_tokens: torch.Tensor,
        fusion_tokens: torch.Tensor,
        field_metadata: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, fields, height, width, dim = field_tokens.shape
        if dim != self.field_dim:
            raise ValueError(f"Expected field dim {self.field_dim}, got {dim}")
        if fusion_tokens.shape[0] != batch or fusion_tokens.shape[2:4] != (
            height,
            width,
        ):
            raise ValueError("Fusion-to-field grids are incompatible")
        latents = fusion_tokens.shape[1]

        field_query = self.field_norm(field_tokens).permute(0, 2, 3, 1, 4)
        field_query = field_query.reshape(batch * height * width, fields, dim)
        fusion_memory = self.fusion_norm(fusion_tokens).permute(0, 2, 3, 1, 4)
        fusion_memory = fusion_memory.reshape(
            batch * height * width, latents, fusion_tokens.shape[-1]
        )
        delta, _ = self.attention(
            field_query, fusion_memory, fusion_memory, need_weights=False
        )
        delta = delta.reshape(batch, height, width, fields, dim).permute(
            0, 3, 1, 2, 4
        )

        if field_metadata.ndim == 2:
            metadata = field_metadata.unsqueeze(0).expand(batch, -1, -1)
        elif field_metadata.ndim == 3 and field_metadata.shape[0] == batch:
            metadata = field_metadata
        else:
            raise ValueError("field_metadata must have shape [M,C] or [B,M,C]")
        metadata = metadata[:, :, None, None].expand(-1, -1, height, width, -1)
        alpha = self._bounded_gate(
            self.gate(torch.cat([field_tokens, delta, metadata], dim=-1))
        )
        updated = field_tokens + alpha * delta
        diagnostics = {
            "fusion_alpha_mean": alpha.mean(),
            "fusion_alpha_min": alpha.min(),
            "fusion_alpha_max": alpha.max(),
        }
        return updated, alpha, diagnostics


class FusionQueryReader(nn.Module):
    def __init__(
        self,
        query_dim: int,
        fusion_dim: int,
        num_heads: int,
        gate_bias: float = -1.0,
        dropout: float = 0.0,
        gate_min: float = 0.0,
        gate_max: float = 1.0,
    ):
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError("query_dim must be divisible by num_heads")
        self.query_dim = int(query_dim)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.query_norm = nn.LayerNorm(query_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=fusion_dim,
            vdim=fusion_dim,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * query_dim),
            nn.Linear(3 * query_dim, query_dim),
            nn.SiLU(),
            nn.Linear(query_dim, 1),
        )
        nn.init.normal_(self.gate[-1].weight, std=0.01)
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def _bounded_gate(self, logits: torch.Tensor) -> torch.Tensor:
        if self.gate_max < self.gate_min:
            raise ValueError("gate_max must be greater than or equal to gate_min")
        span = self.gate_max - self.gate_min
        return self.gate_min + span * logits.sigmoid()

    def forward(
        self,
        field_queries: torch.Tensor,
        fusion_tokens: torch.Tensor,
        query_metadata: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, queries, height, width, dim = field_queries.shape
        if dim != self.query_dim:
            raise ValueError(f"Expected query dim {self.query_dim}, got {dim}")
        if fusion_tokens.shape[0] != batch or fusion_tokens.shape[2:4] != (
            height,
            width,
        ):
            raise ValueError("Fusion query reader grids are incompatible")
        latents = fusion_tokens.shape[1]

        query = self.query_norm(field_queries).permute(0, 2, 3, 1, 4)
        query = query.reshape(batch * height * width, queries, dim)
        memory = self.fusion_norm(fusion_tokens).permute(0, 2, 3, 1, 4)
        memory = memory.reshape(
            batch * height * width, latents, fusion_tokens.shape[-1]
        )
        fusion_queries, _ = self.attention(query, memory, memory, need_weights=False)
        fusion_queries = fusion_queries.reshape(
            batch, height, width, queries, dim
        ).permute(0, 3, 1, 2, 4)

        if query_metadata.ndim == 2:
            metadata = query_metadata.unsqueeze(0).expand(batch, -1, -1)
        elif query_metadata.ndim == 3 and query_metadata.shape[0] == batch:
            metadata = query_metadata
        else:
            raise ValueError("query_metadata must have shape [Q,C] or [B,Q,C]")
        metadata = metadata[:, :, None, None].expand(-1, -1, height, width, -1)
        beta = self._bounded_gate(
            self.gate(torch.cat([field_queries, fusion_queries, metadata], dim=-1))
        )
        output = (1.0 - beta) * field_queries + beta * fusion_queries
        diagnostics = {
            "fusion_beta_mean": beta.mean(),
            "fusion_beta_min": beta.min(),
            "fusion_beta_max": beta.max(),
        }
        return output, beta, diagnostics
