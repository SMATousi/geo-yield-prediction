# Model size: does Yield-ReFiNe depend on its latent count, width or fusion depth?

**Status (2026-10-27): DONE. 455/455 units, 0 failed. No setting changes either metric on average; default kept.
In the paper: supplement `sec:supp_size` / `tab:supp_size`, one line in the method's implementation details.**

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

Parameters (synthetic layout of all streams, `size_smoke` CPU check: every variant builds, trains a step, finite;
the real counts on YieldSAT are larger, see §3):

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

`results/ysat_size_cmp.json` (`rel_cmp.py`, all 7 variants vs the default, seed 0, 2,000 field-season bootstrap
replicates). Real parameter counts (`report.json`, ARG-W fold 0): default 2,041,505; lat4/16/32 2.04M; w64 1,622,561;
w256 3,592,097; d1 1,843,233; d4 2,438,049.

Default, mean over the 4 DEV pairs: CV10 pooled 0.563 / within 0.257; LOYO pooled 0.392 / within 0.169.

Mean change over the 4 pairs, [95% CI]; per pair ARG-W BRA-C GER-R URG-S (* = pair CI excludes 0):

| Variant | CV10 Δ within | CV10 Δ pooled | LOYO Δ within | LOYO Δ pooled | LOYO Δ within per pair |
|---|---|---|---|---|---|
| lat4 | −0.006 [−0.013, +0.001] | +0.000 | +0.011 [−0.004, +0.026] | +0.003 | +0.049 +0.006 +0.005 −0.016* |
| lat16 | +0.004 [−0.005, +0.012] | +0.002 | +0.008 [−0.003, +0.021] | +0.002 | +0.033 +0.004 −0.002 −0.002 |
| lat32 | +0.001 [−0.007, +0.010] | +0.005 | **+0.014 [+0.001, +0.029]** | +0.002 | +0.062* −0.012 +0.012 −0.007* |
| w64 | +0.005 [−0.002, +0.012] | −0.001 | +0.003 [−0.010, +0.018] | +0.009 | +0.017 −0.011 −0.000 +0.006* |
| w256 | −0.007 [−0.015, +0.002] | +0.002 | +0.001 [−0.010, +0.015] | −0.002 | +0.057* −0.023* −0.015 −0.013* |
| d1 | −0.005 [−0.011, +0.002] | +0.004 | +0.007 [−0.004, +0.021] | +0.008 | +0.028 +0.015* −0.012 −0.002 |
| d4 | +0.000 [−0.005, +0.007] | +0.001 | −0.001 [−0.013, +0.013] | −0.009 | +0.024 −0.009 −0.010 −0.007* |

No pooled interval excludes 0. CV10 per pair Δ within: significant only lat4 ARG-W −0.028, d1 ARG-W −0.022,
w64 URG-S +0.004, w256 URG-S −0.006.

- No setting matters on average: |Δ| ≤ 0.007 on CV10 and ≤ 0.014 on LOYO, both metrics. 1 of 28 mean intervals
  excludes 0 (lat32 LOYO within), about what 5% intervals give by chance.
- lat32 LOYO comes from ARG-W (+0.062) with URG-S significantly worse. LOYO ARG-W moves by +0.02 to +0.06 under
  every variant, in both directions of size (default 0.181), so this cell is volatile, not limited by capacity in one
  direction. Not followed up with seeds 1–2: the change is not monotone in any setting.
- Halving the width (1.6M parameters) loses nothing; doubling (3.6M) gains nothing. The default stays.

## 4. Log

- 2026-10-26: design (§1); units and job; CPU smoke of every variant; submitted.
- 2026-10-27: 455/455 done; §3 results; supplement table `supp_size` and text; default kept.
