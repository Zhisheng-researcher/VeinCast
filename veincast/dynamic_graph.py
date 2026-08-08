from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .layers import pad_to_window, window_partition, window_reverse
from .variables import RELATION_GENERIC, RELATION_NAMES


@dataclass
class DynamicGraphState:
    adjacency: torch.Tensor
    allowed_edges: torch.Tensor
    descriptors: torch.Tensor
    relation_context: torch.Tensor
    context_map: torch.Tensor
    residual_mass: torch.Tensor
    entropy: torch.Tensor


class PhysicsGuidedDynamicFieldGraph(nn.Module):
    """
    Build the Physics-Guided Dynamic Field Graph described in VeinCast.

    The graph aggregates compact regional descriptors rather than every spatial token.
    Its context is broadcast back to the window and converted into key adapters, which
    keeps complexity near O(B * R * M^2 * D) instead of O(B * R * M^2 * T * D).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
        relation_ids: torch.Tensor,
        physical_allowed: torch.Tensor,
        topk_residual: int = 4,
        prior_strength: float = 1.0,
        return_adjacency: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = tuple(window_size)
        self.topk_residual = int(topk_residual)
        self.prior_strength = float(prior_strength)
        self.return_adjacency = return_adjacency
        self.num_fields = int(relation_ids.shape[0])
        self.num_relations = max(RELATION_NAMES) + 1

        relation_ids = relation_ids.long()
        physical_allowed = physical_allowed.bool()
        safe_relations = torch.where(
            relation_ids >= 0,
            relation_ids,
            torch.full_like(relation_ids, RELATION_GENERIC),
        )
        self.register_buffer("relation_ids", relation_ids, persistent=True)
        self.register_buffer("safe_relations", safe_relations, persistent=True)
        self.register_buffer("physical_allowed", physical_allowed, persistent=True)

        self.descriptor_norm = nn.LayerNorm(dim)
        self.graph_query = nn.Linear(dim, dim, bias=False)
        self.graph_key = nn.Linear(dim, dim, bias=False)
        self.relation_bias = nn.Embedding(self.num_relations, 1)
        self.relation_values = nn.ModuleList(
            nn.Linear(dim, dim, bias=False) for _ in range(self.num_relations)
        )
        self.adapter = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, num_heads),
        )
        nn.init.constant_(self.gate[-1].bias, 1.0)

    def _dynamic_adjacency(
        self,
        descriptors: torch.Tensor,
        field_present: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # descriptors: [B, R, M, D], scores: [B, R, target, source]
        normalized = self.descriptor_norm(descriptors)
        query = self.graph_query(normalized)
        key = self.graph_key(normalized)
        scores = torch.einsum("brvd,brud->brvu", query, key) * (self.dim ** -0.5)

        relation_bias = self.relation_bias(self.safe_relations).squeeze(-1)
        physical_bias = self.physical_allowed.float() * self.prior_strength
        scores = scores + relation_bias[None, None] + physical_bias[None, None]

        source_available = field_present[:, None, None, :].bool()
        scores = scores.masked_fill(~source_available, -1e4)
        allowed = self.physical_allowed[None, None].expand_as(scores).clone()

        if self.topk_residual > 0:
            residual_scores = scores.masked_fill(
                self.physical_allowed[None, None], -1e4
            )
            topk = min(self.topk_residual, self.num_fields)
            indices = residual_scores.topk(topk, dim=-1).indices
            residual_allowed = torch.zeros_like(allowed)
            residual_allowed.scatter_(-1, indices, True)
            allowed = allowed | residual_allowed

        allowed = allowed & source_available
        scores = scores.masked_fill(~allowed, -1e4)
        adjacency = scores.softmax(dim=-1)
        return adjacency, allowed

    def _relation_context(
        self,
        descriptors: torch.Tensor,
        adjacency: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        context = torch.zeros_like(descriptors)
        for relation_id, projection in enumerate(self.relation_values):
            values = projection(descriptors)
            if relation_id == RELATION_GENERIC:
                edge_mask = allowed & ~self.physical_allowed[None, None]
            else:
                edge_mask = self.relation_ids.eq(relation_id)[None, None] & allowed
            weights = adjacency * edge_mask.to(adjacency.dtype)
            context = context + torch.einsum("brvu,brud->brvd", weights, values)
        return context

    def build_state(
        self,
        tokens: torch.Tensor,
        field_present: torch.Tensor,
    ) -> DynamicGraphState:
        batch, fields, height, width, _ = tokens.shape
        if fields != self.num_fields:
            raise ValueError(f"Expected {self.num_fields} fields, got {fields}")

        padded_tokens, _, _ = pad_to_window(tokens, self.window_size)
        token_windows = window_partition(padded_tokens, self.window_size)
        padded_h, padded_w = padded_tokens.shape[2:4]
        descriptors = token_windows.mean(dim=3).permute(0, 2, 1, 3)

        adjacency, allowed = self._dynamic_adjacency(descriptors, field_present)
        relation_context = self._relation_context(descriptors, adjacency, allowed)
        context_windows = relation_context.permute(0, 2, 1, 3).unsqueeze(3)
        context_windows = context_windows.expand(
            -1, -1, -1, token_windows.shape[3], -1
        )
        context_map = window_reverse(
            context_windows, self.window_size, padded_h, padded_w
        )[:, :, :height, :width]

        nonphysical = ~self.physical_allowed[None, None]
        residual_mass = (
            adjacency * nonphysical.to(adjacency.dtype)
        ).sum(dim=-1).mean()
        entropy = -(
            adjacency.clamp_min(1e-8).log() * adjacency
        ).sum(dim=-1).mean()
        return DynamicGraphState(
            adjacency=adjacency,
            allowed_edges=allowed,
            descriptors=descriptors,
            relation_context=relation_context,
            context_map=context_map,
            residual_mass=residual_mass,
            entropy=entropy,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        static_tokens: torch.Tensor,
        field_present: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch, fields = tokens.shape[:2]
        graph_state = self.build_state(tokens, field_present)
        static = static_tokens[:, None]
        static = static.expand(-1, fields, -1, -1, -1)
        combined = torch.cat([tokens, graph_state.context_map, static], dim=-1)
        key_adapter = self.adapter(combined)
        key_gate = self.gate(combined).sigmoid()

        diagnostics: Dict[str, torch.Tensor] = {
            "graph_residual_mass": graph_state.residual_mass,
            "graph_entropy": graph_state.entropy,
        }
        if self.return_adjacency:
            diagnostics["adjacency"] = graph_state.adjacency.detach()
        return key_adapter, key_gate, diagnostics


# Backward-compatible import name used by early development checkpoints/scripts.
DynamicVariableGraphKeyAdapter = PhysicsGuidedDynamicFieldGraph
