# Specification — Field-Scale Multimodal Geospatial Yield Model

This directory is the authoritative specification for the work built **on top of**
the upstream [MMST-ViT](../README.md) repository (Lin et al., ICCV 2023).

**Update, 2026-09-23:** Phase 0 is complete in a new Conda environment. The CPU
fine-tuning exit command and GPU smoke run succeeded. Phase 1 now has an installable
package and CI configuration. Phase 2 closed the listed correctness defects; the
current contract suite has 52 passing tests and two strict expected failures. The
assessments below and in `status.md` describe the earlier `c8be363`
baseline unless explicitly updated; training still uses synthetic batches. See
[RUNNING.md](../RUNNING.md).

**YieldSAT update, 2026-09-28:** The alternative
[YieldSAT branch](./yieldsat_data_contract.md) (`yieldsat_preprocessed_v1`) is
implemented in point mode and has trained on real data. That covers all four
countries, snapshot-verified sources, grouped/geographic splits, one encoder per
stream and a contract-selected entry point with held-out per-country/per-crop
metrics. See §7 of that document for the step-by-step log, results and open items.
This is the first real-data training in the repository; the downloader/native-grid
branch below is still unimplemented. The suite now has 68 passing tests.

**Full-layer update, 2026-09-28:** The downloader and PlanetScope source inventory
is now captured in [layer_integration.md](./layer_integration.md), with fifteen
implementation tasks mapped to Phases 3–4. These are requirements and planned work,
not a claim that real-data ingestion or full-layer training already runs.

The upstream project predicts **county-level** crop yield from Sentinel-2 imagery +
WRF-HRRR weather using a Multi-Modal / Spatial / Temporal ViT stack. Starting from
that base, 48 commits (2026-09-17 → 2026-09-18) added a largely independent second
system: a **field-scale, native-resolution, multimodal geospatial foundation model**
that predicts dense within-field yield maps and degrades gracefully when input
modalities are missing.

Both systems live in the same tree and share almost nothing but `util/`. Read
[architecture.md](./architecture.md) first if that surprises you.

---

## Documents

| Document | Read it when |
|---|---|
| [mission.md](./mission.md) | You need the goal, scope boundaries, and success criteria. |
| [architecture.md](./architecture.md) | You are writing or reviewing code. Module map, tensor contracts, data flow, and where the two systems divide. §9 gives the exact YieldSAT point-model architecture used for all reported runs. |
| [data_contract.md](./data_contract.md) | Contract-branch index plus existing flat-file demo-loader layout, masks and limitations. |
| [yieldsat_data_contract.md](./yieldsat_data_contract.md) | Alternative YieldSAT contract for Argentina, Brazil, Germany and Uruguay: schemas, country-specific decoding, YS-01–YS-10 tasks, and (§7) the implementation log with audits, splits and real-data results. |
| [yieldsat_paper_comparison.md](./yieldsat_paper_comparison.md) | Plan (PC-01–PC-08) for evaluating on the YieldSAT paper's CV10/LORO/LOYO per-crop protocol, with open protocol questions and reporting rules. |
| [yieldsat_cluster_runs.md](./yieldsat_cluster_runs.md) | Running the full YieldSAT suite on a GPU cluster: suites, plans, jobs (country–crop pair shards), shared data layout, W&B logging, submission and runtime estimate. |
| [layer_integration.md](./layer_integration.md) | Full downloader/PlanetScope inventory, target model contracts, and LI-01–LI-15 compliance tasks. |
| [knowledge_pretraining.md](./knowledge_pretraining.md) | Approved v2 direction for expert-validated relationships constraining sensor embeddings, optional VLM/LLM applicability estimation, and KP-01–KP-08 tasks; planned, not implemented. |
| [status.md](./status.md) | You want the honest state: what is verified, what is orphaned, and the defect list with file:line. |
| [tech_stack.md](./tech_stack.md) | You are provisioning an environment or resolving dependencies. |
| [roadmap.md](./roadmap.md) | You are deciding what to do next. Prioritized, with the blocking order made explicit. |

---

## The one-paragraph status

The **architecture is real and largely complete**; the **pipeline is not**. Every
capability claimed in `.primae/COMPLETENESS_TODO.md` has code behind it, and the
model code is coherent and readable. But the two flagship entry points
(`main_multimodal_finetune.py`, `main_pretrain_multimodal.py`) train on
`torch.randn` synthetic tensors, never on the real `rasterio`-backed loaders that
were written alongside them. Phase 0 smoke runs now succeed, and Phase 1 tests
cover the main System B contracts, but known defects and data integration remain.
Treat this repo as a **validated architectural prototype awaiting a data pipeline**,
not as a trained model. See
[status.md](./status.md) for the evidence.

---

## Conventions

- File references are `path:line` against the tree as of commit `c8be363`.
- "Verified" means read in source or executed. "Claimed" means asserted by a
  commit message, docstring, or `.primae/` receipt but not independently confirmed.
- `.primae/` holds machine-generated build receipts from the generating pipeline.
  They are inputs to this spec, not a substitute for it — `.primae/completeness.json`
  reports 83% requirement coverage, which measures *presence of code*, not
  correctness or integration.
