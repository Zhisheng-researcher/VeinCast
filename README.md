# VeinCast

This repository contains the paper-aligned implementation of **VeinCast**, a
closed-set global medium-range weather forecaster. Each application predicts
the same registered 69 ERA5 fields on a `121 × 240` grid with a shared 6-hour
transition operator.

## Method components

- `PhysicsGuidedDynamicFieldGraph` builds the field graph from physical
  relations and Top-K state-dependent residual edges.
- `GraphConditionedFieldToLatentAttention` and `LatentUFusionBackbone` implement
  graph-conditioned latent fusion with four 384-dimensional latent slots.
- The relation-aware decoder reconstructs the fixed 69-field forecast and
  applies the normalized-space residual update described in the appendix.
- `VeinCastForecastLoss` is the masked latitude-weighted Huber objective with
  threshold `2`. No reconstruction or equation-based physical regularizer is
  included in the reported objective.

See `PAPER_ALIGNMENT.md` for the exact paper-to-code correspondence.

## Installation

```bash
python -m pip install -r requirements.txt
```

Edit the ERA5 and static-data paths in the configuration files before running.
The dynamic NetCDF file must expose the variables and pressure levels defined in
`variables.py`; the canonical registry order contains 69 fields. The optional
surface-pressure field is used to exclude below-ground pressure levels from
supervision.

## Three-stage training

Stage 1 learns one 6-hour transition:

```bash
python train.py --config configs/veincast_stage1.json
```

Stage 2 restarts optimization from the best Stage-1 weights and trains a
two-step rollout:

```bash
python train.py \
  --config configs/veincast_stage2_rollout2.json \
  --init-from artifacts/veincast_stage1/best.pt
```

Stage 3 restarts optimization from the best Stage-2 weights and trains a
four-step rollout:

```bash
python train.py \
  --config configs/veincast_stage3_rollout4.json \
  --init-from artifacts/veincast_stage2_rollout2/best.pt
```

For distributed training, launch the same commands with `torchrun`. Batch size
in the configuration is per process. Rollout steps after the first use a full
availability mask; Stages 2 and 3 use sample-wise teacher forcing with
probability `0.25` and detached predicted inputs.

## Evaluation and forecasting

Evaluate all registered fields at the paper lead times:

```bash
python evaluate.py \
  --checkpoint artifacts/veincast_stage3_rollout4/best.pt \
  --max-lead-hours 336
```

Create a recursive 24-hour forecast:

```bash
python predict.py \
  --checkpoint artifacts/veincast_stage3_rollout4/best.pt \
  --lead-hours 24
```

The prediction archive stores `prediction` with shape
`[forecast_steps, 69, 121, 240]`, together with canonical field labels,
coordinates, and lead hours. VeinCast's reported interface does not interpolate
or predict unregistered pressure levels.

## Compatibility

New code should import `VeinCast` from `veincast.py`. The small `model.py`
wrapper and selected class aliases preserve imports used by early development
scripts; module attribute names are unchanged so existing state dictionaries
remain loadable.
