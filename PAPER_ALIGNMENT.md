# Paper-to-code alignment

| Paper or appendix specification | Implementation |
|---|---|
| Closed-set 69-field transition | `VeinCast.forward` always decodes the canonical registry |
| Input/output grid `121 × 240` | `model.image_size` |
| Patch size `4 × 4` | `model.patch_size` |
| Encoder widths `96/192`, depths `2/4`, heads `4/8` | `embed_dim`, `depths`, `num_heads` |
| Earth window `4 × 8` | `window_size` and `fusion_window_size` |
| Physical graph with Top-K `4` residual edges | `PhysicsGuidedDynamicFieldGraph`, `graph_topk_residual` |
| Physical-prior strength `1.0` | `graph_prior_strength` |
| Four latent slots of width `384` | `fusion_latents`, `fusion_dim` |
| Latent U-Backbone depths `3/4/2`, heads `12/24/12` | `fusion_depths`, `fusion_heads` |
| Bounded feedback gate, maximum `0.25` | `fusion_feedback_gate_max` |
| Bounded decoder fusion gate, maximum `0.5` | `query_fusion_gate_max` |
| Recovery soft clamp `8` | `recovery_output_soft_clamp` |
| Residual scale `0.25`; final soft clamp `12` | `prediction_residual_scale`, `prediction_output_soft_clamp` |
| Base transition `6 h` | `data.base_lead_hours` |
| Masked latitude-weighted Huber, threshold `2` | `VeinCastForecastLoss`, `loss.huber_delta` |
| Progressive rollout `1 → 2 → 4` | the three files in `configs/` |
| Stage-2/3 teacher forcing `0.25` | `teacher_forcing_probability` |
| Detached autoregressive inputs | `detach_rollout: true` |

The released curriculum configurations are:

| Stage | Rollout steps | Epochs | Gradient accumulation | Teacher forcing |
|---|---:|---:|---:|---:|
| 1 | 1 | 150 | 1 | 0.00 |
| 2 | 2 | 10 | 4 | 0.25 |
| 3 | 4 | 10 | 4 | 0.25 |

The code treats graph priors and numerical clamps as inductive biases and
stabilization mechanisms. It does not claim equation-level conservation, and
the reported objective contains no humidity, reconstruction, or other physical
regularization term.
