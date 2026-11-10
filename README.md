# MUPSR

Code for MUPSR, a two-stage price-aware sequential recommendation model with
market-context modeling and user-level price and repeat-purchase components.
The method and implementation are described in [docs/method.md](docs/method.md).

## Repository contents

- `src/mupsr/`: model, data pipeline, training, evaluation, and diagnostics.
- `preparation/` and `scripts/`: data preparation and package utilities.
- `configs/`: main model and ablation configurations.
- `tests/`: synthetic-data tests that do not require the research dataset.
- [docs/FILES.md](docs/FILES.md): a file-by-file guide.

The repository excludes purchase records, prepared caches, checkpoints, and run
outputs. See [LICENSE](LICENSE) for the code license.

## Environment and checks

The code was developed on Linux with Python 3.11, NumPy 1.26.4, PyTorch
2.5.1+cu118, SciPy 1.15.3, and SQLite 3.47.0. A CUDA-capable GPU is needed
for the full training workflow. Create the environment and run the package
checks with:

```bash
conda env create -f environment.yml
conda activate mupsr-r1-b2-b1
bash run.sh verify
bash run.sh test
```

`verify` checks the published files and model configuration. `test` runs the
unit and end-to-end checks on generated synthetic inputs. Neither command
requires the research dataset.

## Data

The experiments use purchase events from the [eCommerce Events History in
Cosmetics Shop](https://www.kaggle.com/mkechinov/ecommerce-events-history-in-cosmetics-Shop)
dataset. This repository does not include the dataset or the frozen split
package. The public raw dataset alone is insufficient to recreate the exact
split and cache identities used by the preparation script.

To run the full workflow, supply the matching split package with these files:

```text
data/source/{train,valid,test,items,categories,users}.csv.gz
data/source/summary.json
data/source/READY
data/expected_prepared_metadata.json
data/expected_event_metadata.json
data/sqlite_layout_delta.json
```

The preparation script validates the inputs and generated artifacts against
recorded hashes. With that package in place, run:

```bash
bash run.sh prepare
```

Generated files are written to `work/prepared/` and `work/events/`.

## Train and evaluate

After preparing the data, run a short workflow check, train the main model,
and evaluate its saved checkpoint:

```bash
bash run.sh smoke runs/smoke
bash run.sh train runs/main
bash run.sh evaluate runs/main/best_stage2.pt runs/main_eval
```

Each output directory must be new. Training uses `configs/main.json` and
selects the checkpoint using validation NDCG@10. For the configuration-only
ablations, use `bash run.sh baseline <output-dir>` and
`bash run.sh recency-baseline <output-dir>`. Run `bash run.sh` to see all
available commands.

The scripts resolve paths relative to the repository. Override their defaults
in the shell when the data or Python interpreter is elsewhere:

```bash
export PYTHON=/path/to/python
export MUPSR_SOURCE=/path/to/split_inputs
export MUPSR_DATA=/path/to/prepared
export MUPSR_CACHE=/path/to/events
```
