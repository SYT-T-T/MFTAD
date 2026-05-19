# MFTAD

MFTAD implements the Multi-View Fuzzy Tree Anomaly Detection pipeline used in the accompanying paper code release. It builds adaptive feature views, constructs fuzzy rough set trees, and fuses multi-granular anomaly scores.

## Requirements

- Python 3.8+
- numpy
- pandas
- torch
- scikit-learn
- numba
- psutil
- mat4py
- matplotlib

Install dependencies with your preferred environment manager, for example:

```bash
pip install numpy pandas torch scikit-learn numba psutil mat4py matplotlib
```

## Repository Structure

- `mg_tbd.py`: Core MFTAD model implementation.
- `mg_tbd_main.py`: Main experiment runner (generates ROC-ready outputs).
- `ablation.py`: Macro and micro ablation experiments.
- `mg_tbd_utils.py`: Data loading, preprocessing, evaluation, and visualization helpers.
- `units.py`: Dataset loading utilities.
- `datasets/`: Example `.npz` and `.mat` datasets.

## Data Format

The loaders expect either:

- `.npz` files containing `X` (features) and `y` (labels), or
- `.mat` files containing `trandata` (last column is label) or `X` and `y`.

Labels are assumed to be binary, where `1` indicates anomalies.

## Quick Start

Run the main evaluation pipeline on all datasets in `datasets/`:

```bash
python mg_tbd_main.py
```

Outputs:

- `result_ROC/ROC_<dataset>.csv` for each dataset.

Run ablation studies:

```bash
python ablation.py
```

Outputs:

- `ablation_macro_results.csv`
- `ablation_micro_results.csv`

## Notes

- Random seeds are fixed in the scripts for reproducibility.
- `best_delta.csv` (optional) can be placed in the repository root for per-dataset delta settings in ablation experiments.
