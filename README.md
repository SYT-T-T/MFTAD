# MFTAD

## Contents

- `reproducibility/relation_mftad.py`: detector implementation.
- `reproducibility/run_paper_experiments.py`: benchmark and analysis entry point.
- `reproducibility/experiment_utils.py`: dataset loading, metrics, statistics, and reporting helpers.
- `reproducibility/tests/`: unit tests for core behavior and result handling.
- `reproducibility/paper_conventions.py`: paper dataset names and abbreviations.
- `tables/table_datasets.tex`: the paper's dataset registry, needed by the convention test.
- Supporting scripts for matrices, tables, statistics, dataset repair, and consistency checks.

## Environment

Python 3.10 or newer is recommended. Install the core dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

Numba is optional and accelerates parts of the detector. `psutil` is optional for process-memory measurements; install both with `python -m pip install -r requirements-optional.txt` when needed. The external comparison methods have additional method-specific dependencies; obtain those implementations from their original repositories and review their licenses separately.

## Data

No datasets are redistributed here. Obtain the benchmark datasets from their original sources and pass their local root with `--dataset-root`. Supply the nominal-feature schema map with `--nominal-map`; use `--require-schema-coverage` for paper-protocol runs. Keep downloaded data outside this repository (the conventional local folders `datasets/` and `data/` are ignored by Git).

The paper's fixed dataset registry is provided under `reproducibility/`. It records dataset names and labels, not the underlying observations.

## Run

Inspect available experiments and options:

```bash
python reproducibility/run_paper_experiments.py --help
python reproducibility/run_paper_experiments.py benchmark --help
```

Run the automated tests:

```bash
python -m unittest discover -s reproducibility/tests -p "test_*.py" -v
```

For a paper reproduction, use the exact command and protocol documented with the corresponding paper artifact. Do not use `--resume` to combine outputs produced by different code, data, or configuration versions. Store generated results outside the source tree or in an ignored local output directory.

## Contact
If you have any questions, please contact 17779191866@163.com.
