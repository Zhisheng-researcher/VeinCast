from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_graph import DynamicGraphState, PhysicsGuidedDynamicFieldGraph
from .fusion import (
    GraphConditionedFieldToLatentAttention,
    FusionLatentInitializer,
    FusionQueryReader,
    FusionToFieldFeedback,
    LatentUFusionBackbone,
)
from .layers import (
    ContinuousFourierEmbedding,
    EarthWindowBlock,
    FieldMetadataEmbedding,
    FieldPatchRecovery,
    PatchExpand,
    PatchMerging,
    QuerySpatialBlock,
    StaticPatchEmbedding,
    VariableCrossAttention,
)
from .stems import DualModalityStem
from .variables import VariableRegistry, build_query_prior


class VeinCast(nn.Module):
    """
    VeinCast closed-set global medium-range weather forecaster.

    The reported model predicts the registered 69-field state with a shared
    6-hour transition operator. Query metadata remains explicit inside the
    relation-aware decoder, but the public forecasting workflow uses the fixed
    registry described in the paper.
    """

    def __init__(self, registry: VariableRegistry, model_config: Dict[str, object]):
        super().__init__()
        self.registry = registry
        self.config = dict(model_config)
        self.image_size = tuple(model_config["image_size"])
        self.patch_size = tuple(model_config["patch_size"])
        self.embed_dim = int(model_config["embed_dim"])
        self.depths = tuple(model_config["depths"])
        self.num_heads = tuple(model_config["num_heads"])
        self.window_size = tuple(model_config["window_size"])
        self.static_channels = int(model_config["static_channels"])
        self.low_dim = 2 * self.embed_dim

        high_grid = (
            math.ceil(self.image_size[0] / self.patch_size[0]),
            math.ceil(self.image_size[1] / self.patch_size[1]),
        )
        low_grid = (math.ceil(high_grid[0] / 2), math.ceil(high_grid[1] / 2))
        self.high_grid = high_grid
        self.low_grid = low_grid

        variable_ids, pressures, surface = registry.query_arrays()
        self.register_buffer(
            "field_variable_ids", torch.from_numpy(variable_ids), persistent=True
        )
        self.register_buffer(
            "field_pressures", torch.from_numpy(pressures), persistent=True
        )
        self.register_buffer(
            "field_surface", torch.from_numpy(surface), persistent=True
        )
        relation_ids, physical_allowed = registry.physical_relations()
        relation_ids_tensor = torch.from_numpy(relation_ids)
        physical_allowed_tensor = torch.from_numpy(physical_allowed)

        pressure_bands = int(model_config.get("pressure_fourier_bands", 8))
        dropout = float(model_config.get("dropout", 0.0))
        drop_path_max = float(model_config.get("drop_path", 0.1))
        mlp_ratio = float(model_config.get("mlp_ratio", 4.0))
        topk = int(model_config.get("graph_topk_residual", 4))
        prior_strength = float(model_config.get("graph_prior_strength", 1.0))

        stem_depth = int(model_config.get("stem_refinement_depth", 1))
        self.input_stem = DualModalityStem(
            registry=registry,
            dim=self.embed_dim,
            patch_size=self.patch_size,
            pressure_bands=pressure_bands,
            refinement_depth=stem_depth,
            surface_depth=model_config.get("surface_stem_depth"),
            atmosphere_depth=model_config.get("atmosphere_stem_depth"),
        )
        self.high_query_metadata = FieldMetadataEmbedding(
            registry.num_variables, self.embed_dim, pressure_bands
        )
        self.static_embedding = StaticPatchEmbedding(
            self.static_channels, self.embed_dim, self.patch_size
        )
        self.static_low_projection = nn.Linear(self.embed_dim, self.low_dim)

        total_encoder_blocks = sum(self.depths)
        drop_paths = torch.linspace(0.0, drop_path_max, total_encoder_blocks).tolist()
        self.high_graphs = nn.ModuleList()
        self.high_blocks = nn.ModuleList()
        for index in range(self.depths[0]):
            self.high_graphs.append(
                PhysicsGuidedDynamicFieldGraph(
                    dim=self.embed_dim,
                    num_heads=self.num_heads[0],
                    window_size=self.window_size,
                    relation_ids=relation_ids_tensor,
                    physical_allowed=physical_allowed_tensor,
                    topk_residual=topk,
                    prior_strength=prior_strength,
                )
            )
            self.high_blocks.append(
                EarthWindowBlock(
                    dim=self.embed_dim,
                    grid_size=high_grid,
                    num_heads=self.num_heads[0],
                    window_size=self.window_size,
                    shifted=index % 2 == 1,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_paths[index],
                )
            )

        self.patch_merging = PatchMerging(self.embed_dim)
        self.low_graphs = nn.ModuleList()
        self.low_blocks = nn.ModuleList()
        offset = self.depths[0]
        for index in range(self.depths[1]):
            self.low_graphs.append(
                PhysicsGuidedDynamicFieldGraph(
                    dim=self.low_dim,
                    num_heads=self.num_heads[1],
                    window_size=self.window_size,
                    relation_ids=relation_ids_tensor,
                    physical_allowed=physical_allowed_tensor,
                    topk_residual=topk,
                    prior_strength=prior_strength,
                )
            )
            self.low_blocks.append(
                EarthWindowBlock(
                    dim=self.low_dim,
                    grid_size=low_grid,
                    num_heads=self.num_heads[1],
                    window_size=self.window_size,
                    shifted=index % 2 == 1,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_paths[offset + index],
                )
            )

        if not self.low_graphs:
            raise ValueError("The low-resolution encoder must contain at least one block")

        fusion_dim = int(model_config.get("fusion_dim", self.low_dim))
        fusion_latents = int(model_config.get("fusion_latents", 4))
        fusion_depths = tuple(model_config.get("fusion_depths", [1, 1, 1]))
        default_fusion_heads = [
            self.num_heads[1],
            2 * self.num_heads[1],
            self.num_heads[1],
        ]
        fusion_heads = tuple(
            model_config.get("fusion_heads", default_fusion_heads)
        )
        fusion_window = tuple(
            model_config.get("fusion_window_size", self.window_size)
        )
        fusion_drop_path = float(
            model_config.get("fusion_drop_path", drop_path_max)
        )
        calendar_bands = int(model_config.get("calendar_fourier_bands", 8))
        geo_bands = int(model_config.get("geo_fourier_bands", 8))
        self.fusion_initializer = FusionLatentInitializer(
            fusion_dim=fusion_dim,
            num_latents=fusion_latents,
            grid_size=low_grid,
            static_dim=self.low_dim,
            lead_bands=pressure_bands,
            calendar_bands=calendar_bands,
            geo_bands=geo_bands,
            use_calendar_embedding=bool(
                model_config.get("use_calendar_embedding", True)
            ),
        )
        self.field_to_fusion = GraphConditionedFieldToLatentAttention(
            field_dim=self.low_dim,
            fusion_dim=fusion_dim,
            num_heads=fusion_heads[0],
            num_latents=fusion_latents,
            window_size=self.window_size,
            dropout=dropout,
            centrality_floor=float(model_config.get("fusion_centrality_floor", 1e-8)),
        )
        self.fusion_backbone = LatentUFusionBackbone(
            dim=fusion_dim,
            grid_size=low_grid,
            num_latents=fusion_latents,
            depths=fusion_depths,
            num_heads=fusion_heads,
            window_size=fusion_window,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=fusion_drop_path,
        )
        self.fusion_feedback = FusionToFieldFeedback(
            field_dim=self.low_dim,
            fusion_dim=fusion_dim,
            num_heads=self.num_heads[1],
            gate_bias=float(model_config.get("fusion_feedback_gate_bias", -2.0)),
            dropout=dropout,
            gate_min=float(model_config.get("fusion_feedback_gate_min", 0.0)),
            gate_max=float(model_config.get("fusion_feedback_gate_max", 1.0)),
        )
        self.fusion_query_reader = FusionQueryReader(
            query_dim=self.low_dim,
            fusion_dim=fusion_dim,
            num_heads=self.num_heads[1],
            gate_bias=float(model_config.get("query_fusion_gate_bias", -1.0)),
            dropout=dropout,
            gate_min=float(model_config.get("query_fusion_gate_min", 0.0)),
            gate_max=float(model_config.get("query_fusion_gate_max", 1.0)),
        )
        self.low_query_metadata = FieldMetadataEmbedding(
            registry.num_variables, self.low_dim, pressure_bands
        )
        self.high_lead_embedding = ContinuousFourierEmbedding(
            self.embed_dim, pressure_bands
        )
        self.low_lead_embedding = ContinuousFourierEmbedding(
            self.low_dim, pressure_bands
        )
        self.low_query_attention = VariableCrossAttention(
            self.low_dim, self.num_heads[1], dropout=dropout
        )
        self.query_expand = PatchExpand(self.low_dim, self.embed_dim)
        self.high_query_attention = VariableCrossAttention(
            self.embed_dim, self.num_heads[0], dropout=dropout
        )
        self.query_fusion = nn.Sequential(
            nn.LayerNorm(2 * self.embed_dim),
            nn.Linear(2 * self.embed_dim, self.embed_dim),
            nn.GELU(),
        )
        self.query_blocks = nn.ModuleList(
            QuerySpatialBlock(
                dim=self.embed_dim,
                grid_size=high_grid,
                num_heads=self.num_heads[0],
                window_size=self.window_size,
                shifted=index % 2 == 1,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=drop_path_max * 0.5,
            )
            for index in range(2)
        )
        self.recovery = FieldPatchRecovery(
            self.embed_dim,
            registry.num_variables,
            self.patch_size,
            self.image_size,
            bounded_affine=bool(model_config.get("bounded_recovery_affine", False)),
            affine_scale_radius=float(
                model_config.get("recovery_affine_scale_radius", 0.5)
            ),
            affine_bias_radius=float(
                model_config.get("recovery_affine_bias_radius", 0.5)
            ),
            output_soft_clamp=float(
                model_config.get("recovery_output_soft_clamp", 0.0)
            ),
            zero_init=bool(model_config.get("recovery_zero_init", False)),
            init_std=float(model_config.get("recovery_init_std", 0.0)),
        )
        self.residual_prediction = bool(model_config.get("residual_prediction", False))
        self.prediction_residual_scale = float(
            model_config.get("prediction_residual_scale", 1.0)
        )
        self.prediction_output_soft_clamp = float(
            model_config.get("prediction_output_soft_clamp", 0.0)
        )

        default_bias = build_query_prior(registry, variable_ids, pressures, surface)
        self.register_buffer(
            "default_query_bias", torch.from_numpy(default_bias), persistent=True
        )

    def _query_metadata(
        self,
        module: FieldMetadataEmbedding,
        variable_ids: torch.Tensor,
        pressures: torch.Tensor,
        surface: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        metadata = module(variable_ids, pressures, surface)
        if metadata.ndim == 2:
            metadata = metadata.unsqueeze(0).expand(batch_size, -1, -1)
        return metadata

    def _query_bias(
        self,
        variable_ids: torch.Tensor,
        pressures: torch.Tensor,
        surface: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            variable_ids.ndim == 1
            and variable_ids.shape == self.field_variable_ids.shape
            and torch.equal(variable_ids.cpu(), self.field_variable_ids.cpu())
            and torch.allclose(pressures.cpu(), self.field_pressures.cpu())
        ):
            return self.default_query_bias.to(device=device, dtype=dtype)

        if variable_ids.shape != pressures.shape or variable_ids.shape != surface.shape:
            raise ValueError("Query variable, pressure and surface tensors must match")
        if variable_ids.ndim == 1:
            metadata_rows = [(variable_ids, pressures, surface)]
        elif variable_ids.ndim == 2:
            metadata_rows = list(zip(variable_ids, pressures, surface))
        else:
            raise ValueError("Query metadata must have shape [query] or [batch, query]")

        bias_rows = [
            build_query_prior(
                self.registry,
                ids.detach().cpu().numpy(),
                pressure.detach().cpu().numpy(),
                is_surface.detach().cpu().numpy(),
            )
            for ids, pressure, is_surface in metadata_rows
        ]
        bias = bias_rows[0] if variable_ids.ndim == 1 else np.stack(bias_rows)
        return torch.as_tensor(bias, device=device, dtype=dtype)

    def _soft_clamp(self, values: torch.Tensor, limit: float) -> torch.Tensor:
        if limit <= 0:
            return values
        limit_tensor = values.new_tensor(limit)
        return limit_tensor * torch.tanh(values / limit_tensor)

    def _apply_prediction_constraints(
        self,
        decoded: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        prediction = decoded
        if self.residual_prediction:
            prediction = state + self.prediction_residual_scale * decoded
        return self._soft_clamp(prediction, self.prediction_output_soft_clamp)

    def encode(
        self,
        state: torch.Tensor,
        static: torch.Tensor,
        input_present: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        DynamicGraphState,
        Dict[str, torch.Tensor],
    ]:
        variable_ids = self.field_variable_ids
        pressures = self.field_pressures
        surface = self.field_surface
        high, grid = self.input_stem(
            state, variable_ids, pressures, surface, input_present
        )
        if grid != self.high_grid:
            raise ValueError(
                f"Configured image_size produces {self.high_grid}, but input produced {grid}"
            )
        static_high = self.static_embedding(static)
        high = high + static_high[:, None]

        diagnostics = []
        for graph, block in zip(self.high_graphs, self.high_blocks):
            adapter, gate, graph_aux = graph(high, static_high, input_present)
            high = block(high, adapter, gate)
            diagnostics.append(graph_aux)

        low = self.patch_merging(high)
        static_image = static_high.permute(0, 3, 1, 2)
        static_low = F.interpolate(
            static_image, size=low.shape[2:4], mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        static_low = self.static_low_projection(static_low)
        low = low + static_low[:, None]

        for graph, block in zip(self.low_graphs, self.low_blocks):
            adapter, gate, graph_aux = graph(low, static_low, input_present)
            low = block(low, adapter, gate)
            diagnostics.append(graph_aux)

        final_graph_state = self.low_graphs[-1].build_state(low, input_present)
        diagnostics.append(
            {
                "graph_residual_mass": final_graph_state.residual_mass,
                "graph_entropy": final_graph_state.entropy,
            }
        )

        aux = {
            "graph_residual_mass": torch.stack(
                [item["graph_residual_mass"] for item in diagnostics]
            ).mean(),
            "graph_entropy": torch.stack(
                [item["graph_entropy"] for item in diagnostics]
            ).mean(),
        }
        return high, low, static_low, final_graph_state, aux

    def fuse(
        self,
        low: torch.Tensor,
        static_low: torch.Tensor,
        graph_state: DynamicGraphState,
        input_present: torch.Tensor,
        lead_hours: torch.Tensor,
        calendar_features: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        latents = self.fusion_initializer(
            static_low, lead_hours, calendar_features
        )
        fusion, attention_aux = self.field_to_fusion(
            latents, low, graph_state, input_present
        )
        fusion = self.fusion_backbone(fusion)
        field_metadata = self.low_query_metadata(
            self.field_variable_ids,
            self.field_pressures,
            self.field_surface,
        )
        low_feedback, _, feedback_aux = self.fusion_feedback(
            low, fusion, field_metadata
        )
        aux = {
            **attention_aux,
            **feedback_aux,
            "fusion_latent_norm": fusion.square().mean().sqrt(),
        }
        return low_feedback, fusion, aux

    def decode(
        self,
        high: torch.Tensor,
        low: torch.Tensor,
        fusion: torch.Tensor,
        input_present: torch.Tensor,
        lead_hours: torch.Tensor,
        query_variable_ids: torch.Tensor,
        query_pressures: torch.Tensor,
        query_surface: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = high.shape[0]
        lead_coordinate = lead_hours.float().reshape(batch) / 24.0
        query_bias = self._query_bias(
            query_variable_ids, query_pressures, query_surface, high.dtype, high.device
        )

        low_metadata = self._query_metadata(
            self.low_query_metadata,
            query_variable_ids,
            query_pressures,
            query_surface,
            batch,
        )
        low_metadata = low_metadata + self.low_lead_embedding(lead_coordinate)[:, None]
        query_low = self.low_query_attention(
            low, low_metadata, input_present, query_bias
        )
        query_low, _, fusion_query_aux = self.fusion_query_reader(
            query_low, fusion, low_metadata
        )

        query_up = self.query_expand(query_low, self.high_grid)
        high_metadata = self._query_metadata(
            self.high_query_metadata,
            query_variable_ids,
            query_pressures,
            query_surface,
            batch,
        )
        high_metadata = high_metadata + self.high_lead_embedding(lead_coordinate)[:, None]
        query_high = self.high_query_attention(
            high, high_metadata, input_present, query_bias
        )
        query_high = self.query_fusion(torch.cat([query_high, query_up], dim=-1))
        for block in self.query_blocks:
            query_high = block(query_high)

        return self.recovery(query_high, query_variable_ids), fusion_query_aux

    def default_queries(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.field_variable_ids.to(device),
            self.field_pressures.to(device),
            self.field_surface.to(device),
        )

    def forward(
        self,
        state: torch.Tensor,
        static: torch.Tensor,
        input_present: Optional[torch.Tensor] = None,
        lead_hours: Optional[torch.Tensor] = None,
        calendar_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        batch, fields = state.shape[:2]
        if fields != self.registry.num_fields:
            raise ValueError(
                f"Model registry expects {self.registry.num_fields} fields, got {fields}"
            )
        if input_present is None:
            input_present = torch.ones(batch, fields, device=state.device, dtype=state.dtype)
        if lead_hours is None:
            lead_hours = torch.full((batch,), 6.0, device=state.device, dtype=state.dtype)
        query_variable_ids, query_pressures, query_surface = self.default_queries(
            state.device
        )

        high, low, static_low, graph_state, aux = self.encode(
            state, static, input_present
        )
        forecast_low, forecast_fusion, fusion_aux = self.fuse(
            low,
            static_low,
            graph_state,
            input_present,
            lead_hours,
            calendar_features,
        )
        prediction, query_aux = self.decode(
            high,
            forecast_low,
            forecast_fusion,
            input_present,
            lead_hours,
            query_variable_ids,
            query_pressures,
            query_surface,
        )
        prediction = self._apply_prediction_constraints(
            prediction,
            state,
        )
        aux = {**aux, **fusion_aux, **query_aux}
        return {"prediction": prediction, "aux": aux}
