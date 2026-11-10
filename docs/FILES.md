# Repository layout

The public GitHub copy excludes `data/`, including the purchase-split inputs and generated
metadata. The paths below describe the complete experiment package needed for a full rebuild.

The model is implemented once, in the `mupsr` package under `src/`. The final method is R1+B1+B2
(`configs/main.json`); the two shipped ablations restrict it to R1 and R1+B2 through configuration flags
only, so there is a single model engine and a single training entrypoint.

| Path | Contents |
|---|---|
| `src/mupsr/model.py` | The MUPSR model: shared encoders, stage-1 category panel, price-parameter head, stage-2 user encoder with the query-relative recency readout and the state-dependence term, the identification normalization, and the tiled catalogue likelihood. One `MUPSR` class; `use_recency` / `use_repeat` and the nested-model flags select its components. |
| `src/mupsr/data.py` | Runtime data pipeline: integrity-checked prepared cache, frozen candidate protocol, event-level inputs, stage-1 day tables, and the stage-2 batch builders (each query carrying its prediction time and experience entries). |
| `src/mupsr/features.py` | NumPy preprocessing: the user-item experience store (`RepeatIndex`), the Poisson fixed-effect initializer (`fe_init`), and a time-decayed popularity baseline used only by the diagnostics. |
| `src/mupsr/train.py` | Training and full-catalogue evaluation: grouped-learning-rate optimizer, warmup-cosine schedule, weight averaging, the two-stage loop, the candidate-pool filter, and the stage-transfer check. |
| `src/mupsr/evaluate.py` | Re-evaluate a saved checkpoint under its own frozen config and data identities. |
| `src/mupsr/metrics.py` | Full-catalogue, tie-aware R@k / N@k. |
| `src/mupsr/diagnostics.py` | Stage-0 data diagnostics and protocol-faithful reference baselines. |
| `src/mupsr/significance.py` | Paired per-query comparison with bootstrap, paired-t and Wilcoxon, optionally by stratum. |
| `src/mupsr/utils.py` | File hashing, JSON and checkpoint I/O. |
| `configs/main.json` | The final R1+B1+B2 configuration (`use_recency=true`, `use_repeat=true`). |
| `configs/ablation_recency_only.json` | R1+B2 ablation (`use_repeat=false`). |
| `configs/ablation_training_only.json` | R1 ablation (`use_recency=false`, `use_repeat=false`). |
| `preparation/` | Offline rebuild of the prepared and event caches from the frozen split CSVs. Standalone from the runtime package. |
| `scripts/prepare_data.py` | Portable rebuild entrypoint: verify all content, then canonicalize metadata order. |
| `scripts/smoke.py` | Real-data GPU preflight: frozen protocol, one update per stage, transfer and an 8-query evaluation. |
| `scripts/accept_experiment.py` | Validation-only retention check between two runs. |
| `scripts/verify_package.py` | Archive manifest, source hashes and configuration identity. |
| `scripts/env.sh`, `run.sh` | Portable paths, interpreter, threads and a single command entrypoint. |
| `tests/synthetic.py` | Tiny synthetic dataset in the on-disk format of the real prepared cache plus the three split CSVs. |
| `tests/test_core.py` | Structural tests of the model and an end-to-end CPU train/evaluate. |
| `tests/test_model.py` | Tests of the optimized estimation and the state-dependence terms (T1-T13, candidate-pool guard, nested flags, end-to-end). |
| `data/source/{train,valid,test}.csv.gz` | The exact purchase-split inputs; hashed per file when rebuilding. |
| `data/source/{items,categories,users}.csv.gz`, `summary.json`, `READY` | Mappings and data identity the preparation reads and hashes. |
| `data/expected_prepared_metadata.json`, `data/expected_event_metadata.json` | Content and metadata-order references that keep the candidate and event fingerprints. |
| `data/sqlite_layout_delta.json` | Physical-layout delta restoring the historical SQLite byte identity from an SQLite 3.47.0 rebuild. |
| `README.md`, `docs/method.md`, `docs/FILES.md`, `provenance.json` | Operating steps, the final method specification, this guide, and the recorded source, hashes and reference metrics. |

The public repository does not carry the frozen purchase-split inputs or their mappings, checkpoints,
training logs, per-epoch metrics, prediction or cache arrays, the SQLite cache, virtual environments,
or `__pycache__`. Rebuilding the original experiment requires the source inputs with the recorded hashes.
