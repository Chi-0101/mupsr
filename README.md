# MUPSR

This public repository contains the model, preprocessing code, configurations, tests, and recorded
experiment metadata. The purchase-split files and generated caches are excluded. To rebuild the
experiment, place the eight required input files under `data/source/` as described below.
`bash run.sh verify` checks the published code package; `bash run.sh prepare` additionally requires
the excluded input files.

Mental-accounting two-stage price-aware sequential recommendation, with reproducible training on the
Kaggle *eCommerce Events History in Cosmetics Shop* dataset (2019-10 to 2020-02, purchase events only).

The method is described in [`docs/method.md`](docs/method.md). The final configuration
(`configs/main.json`) combines the optimized estimation and training (fixed-effect initialization, a
full first-stage schedule, grouped learning rates, a warmup-cosine schedule, weight averaging, and
identification normalization) with a state-dependence term for repeat purchases and a query-relative
recency bias in the readout. Two shipped ablations restrict the same model to the training-only and the
training-plus-recency variants through configuration flags only.

## Reference results (seed 2026)

| Split | R@5 | N@5 | R@10 | N@10 | R@20 | N@20 |
|---|---:|---:|---:|---:|---:|---:|
| Valid | 0.198445 | 0.144131 | 0.259177 | 0.163746 | 0.328814 | 0.181285 |
| Test  | 0.171432 | 0.125350 | 0.226346 | 0.143093 | 0.289611 | 0.159039 |

Evaluation protocol: per-user leave-last-out; scoring over each day's price-eligible candidate set;
tie-aware Recall/NDCG; candidate fingerprint `d3cf4475…05d1ac`; 58,273 validation and 56,524 test queries.

## Environment

Tested on Linux x86_64, Python 3.11.16, SQLite 3.47.0, NumPy 1.26.4, PyTorch 2.5.1+cu118, SciPy 1.15.3,
Tesla V100 16 GB. Install a CUDA 11.8 driver, then the packages:

```bash
conda env create -f environment.yml
conda activate mupsr-r1-b2-b1
# or, in an existing Python 3.11 / SQLite 3.47.0 environment:
# python -m pip install -r requirements.txt
bash run.sh verify
bash run.sh test
```

`verify` checks the file manifest, the source hashes and the exact configuration identity. `test` runs the
core and optimization test suites (synthetic data is created and cleaned up automatically).

## Rebuild the data

```bash
bash run.sh prepare
```

Inputs are the eight files under `data/source/`; outputs go to `work/prepared/` and `work/events/`. The
rebuild checks every generated array and SQLite file against the reference SHA-256 before canonicalizing
metadata order. Expected results: candidate fingerprint
`d3cf4475612945979d03e51ef53f0b6ae2efa59ae09085a79f39958cbe05d1ac`, event fingerprint
`f2f36f5fed7e43666c3a9cd7206cae6684d1e7c279a9eab0b6f626bc414ede89`, 58,273 validation and 56,524 test
queries.

## Train and evaluate

```bash
bash run.sh smoke runs/smoke
bash run.sh train runs/main_seed2026 2026 cuda
bash run.sh evaluate runs/main_seed2026/best_stage2.pt runs/main_eval cuda
```

Run the smoke preflight first; it uses the real directories and model size, updates each stage once, and
evaluates a few queries for flow checking only. Training runs two full stages: stage 1 up to 40 epochs
(evaluated every 4, early-stopped after 3 non-improving checks); stage 2 up to 40 epochs (early-stopped
after 6), with EMA validation and model selection; model selection uses validation N@10 only. The output
directory must not exist. A completed run contains `environment.json`, `transfer.json`,
`training_log.json`, `best_stage1.pt`, `best_stage2.pt`, `metrics.json` and two `per_query_N10_*.npy`.

## Ablations, diagnostics and acceptance

```bash
bash run.sh baseline runs/R1_seed2026 2026 cuda            # training only
bash run.sh recency-baseline runs/R1_B2_seed2026 2026 cuda # training + recency
bash run.sh diagnose runs/diagnostics
bash run.sh compare runs/R1_seed2026 runs/main_seed2026 runs/diagnostics runs/acceptance_valid.json
```

The acceptance check applies the retention rule on validation only (relative N@10 gain, paired-t p-value,
no stratum declining by more than 3%) and reports bootstrap intervals and the Wilcoxon test; exit code 3
means the retention condition was not met, and the report is still written.

## Paths

Scripts resolve the repository from their own location. The data paths, interpreter and thread count are
set in `scripts/env.sh` and can be overridden:

```bash
export PYTHON=/path/to/python
export MUPSR_SOURCE=/path/to/exact_split_inputs
export MUPSR_DATA=/path/to/prepared
export MUPSR_CACHE=/path/to/events
```

`MUPSR_SOURCE` must match the reference input hashes. The seed is fixed to 2026; different CUDA, drivers or
hardware can produce last-digit floating-point and trajectory differences. See [`docs/FILES.md`](docs/FILES.md)
for the role of each file and `provenance.json` for the recorded source, hashes and reference metrics.
