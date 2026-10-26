# Model size: does Yield-ReFiNe depend on its latent count, width or fusion depth?

**Status (2026-10-26): 455 units submitted (job `smatousi-yieldsat-protocol-size`).**

Related:
- [yieldsat_data_contract.md](./yieldsat_data_contract.md) (2026-09-29 fusion ablation): perceiver_summary chosen over
  perceiver_tokens, token_transformer and concat_mlp; a latent sweep {8, 16, 32} was run only for perceiver_tokens,
  one seed, pixel RMSE on the old farm / block splits, before field context and the relation loss.
- [yieldsat-field-context.md](./yieldsat-field-context.md): the final model `ours-p3nbr-dense-fc-rel10` and its DEV runs.

## 1. Question and design (project lead, 2026-10-26)

The final model was never varied in size on the paper protocol. A reviewer will ask whether its numbers depend on
the chosen capacity. One setting at a time around the default:

| Setting | Flag | Default | Variants | Tag suffix |
|---|---|---|---|---|
| Latent queries of the Perceiver fusion | `--num_latents` | 8 | 4, 16, 32 | `-lat4`, `-lat16`, `-lat32` |
| Token width (stream encoder outputs, fusion, head) | `--embed_dim` | 128 | 64, 256 | `-w64`, `-w256` |
| Latent self-attention blocks | `--depth` | 2 | 1, 4 | `-d1`, `-d4` |

- Pairs: the 4 DEV pairs (ARG-W, BRA-C, GER-R, URG-S), CV10 + LOYO (65 units per variant), seed 0.
- Baseline: the final model's own DEV runs (`protocol_fc_dev_units.json`, seed 0); identical arguments plus the one flag.
- Seed 0 only: on YieldSAT the seed SD of within-field R² is 0.004–0.010 per pair–protocol cell; a variant whose
  difference is near that gets seeds 1–2 afterwards.
- Width does not change the temporal encoders' inner width (`temporal_hidden` 128) or the static encoders' (64);
  it sets the token dimension every stream is projected to, the fusion and the head.
- 4 heads throughout (head dim 16 / 32 / 64 at width 64 / 128 / 256). Heads are not swept.

Parameters (synthetic layout of all streams, `size_smoke` CPU check: every variant builds, trains a step, finite):

| Variant | Parameters | Fusion parameters |
|---|---|---|
| default | 1,127,937 | 465,472 |
| lat4 / lat16 / lat32 | 1,127,425 / 1,128,961 / 1,131,009 | 464,960 / 466,496 / 468,544 |
| w64 / w256 | 738,305 / 2,619,905 | 117,952 / 1,848,640 |
| d1 / d4 | 929,665 / 1,524,481 | 267,200 / 862,016 |

## 2. Runs

`yieldsat_protocol_runs.py units --batch size-dev` → `cluster/tabm/protocol_size_dev_units.json`: 455 units, outputs
`paper/<group>/s2_adm/ours-p3nbr-dense-fc-rel10-<suffix>_seed0/<pair>/foldNN`. Job
`cluster/nautilus/protocol_size_dev_job.yaml`: 30 pods × 4 runs (24 GB cards), 32 Gi, as the fc-dev job.

Comparison per variant: `REL_SUB=paper REL_BASE=ours-p3nbr-dense-fc-rel10 python results/rel_cmp.py
ours-p3nbr-dense-fc-rel10-<suffix> ARG-W,BRA-C,GER-R,URG-S`.

## 3. Results

Pending.

## 4. Log

- 2026-10-26: design (§1); units and job; CPU smoke of every variant; submitted.
