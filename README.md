# MLOpTune

MLOpTune is a lightweight fine-tuning framework for scikit-learn-compatible
models. Describe an experiment in one configuration object, call `run`, and get
back a tuned, refitted model together with an honest test score, the Optuna
study behind it, and the seed that reproduces all of it.

- **Author:** Ziyuan Huang
- **Affiliation:** Microbiology & Microbiome Dynamic AI Hub, University of
  Massachusetts, Worcester

## Features

- **One call, whole workflow.** Stratified train / validation / test split,
  Optuna study on the validation split, refit of the best hyperparameters on
  train + validation, one final score on the untouched test split.
- **Reproducible by name.** The `experiment_name` is hashed into the seed for
  the splits, the sampler and every estimator that accepts `random_state`.
  The same name gives the same run on any machine or operating system.
- **Any scikit-learn classifier or regressor**, addressed by class name, plus
  `XGBClassifier` and `XGBRegressor` when `xgboost` is installed.
- **Any scikit-learn scorer** by name (`roc_auc`, `f1_macro`,
  `neg_root_mean_squared_error`, ...) or as a `make_scorer` callable.
- **Fail fast.** The config validates itself, model names are checked against
  the problem type, and hyperparameter names are checked against the
  estimator's constructor before any training starts.
- **Fail soft during the study.** A hyperparameter combination that raises
  costs one failed Optuna trial, not the run.
- **Works with NumPy arrays, pandas DataFrames, lists and scipy sparse
  matrices.**

## Installation

From PyPI:

```bash
pip install mloptune
pip install "mloptune[test]"   # plus pytest
```

The latest development version straight from GitHub (no clone needed):

```bash
pip install "git+https://github.com/melhzy/MLOpTune.git"
pip install "mloptune[test] @ git+https://github.com/melhzy/MLOpTune.git"   # plus pytest
```

Or from a local checkout, for development:

```bash
git clone https://github.com/melhzy/MLOpTune.git
cd MLOpTune
pip install -e .[test]
```

Requires Python 3.10 or newer. On macOS, XGBoost needs the OpenMP runtime
(`brew install libomp`); without it the XGBoost aliases are simply absent and
everything else works.

## Quick start

```python
from sklearn.datasets import load_iris

from mloptune import FineTuneConfig, FineTuner

dataset = load_iris()
config = FineTuneConfig(
    model_name="RandomForestClassifier",
    problem_type="classification",
    experiment_name="iris-demo",
    search_space={
        "n_estimators": {"type": "int", "low": 10, "high": 30},
        "max_depth": {"type": "int", "low": 2, "high": 5},
    },
    n_trials=3,
)

result = FineTuner(config).run(dataset.data, dataset.target)
print(result.best_params)
print(result.test_score)
print(result.study.trials_dataframe())  # every configuration that was evaluated
```

### Configuration reference

| Field | Default | Meaning |
|---|---|---|
| `model_name` | required | scikit-learn class name, or `XGBClassifier` / `XGBRegressor` |
| `problem_type` | required | `"classification"` or `"regression"`; the model must match |
| `experiment_name` | required | Free text; hashed into the seed |
| `search_space` | `{}` | Hyperparameters to tune (see below); empty means fit `model_kwargs` as-is |
| `n_trials` | `10` | Number of Optuna trials |
| `n_startup_trials` | Optuna default (10) | Random trials before TPE starts modelling |
| `model_kwargs` | `{}` | Fixed constructor arguments for every candidate |
| `scoring` | `accuracy` / `neg_mean_squared_error` | Any scikit-learn scorer name or callable |
| `test_size`, `validation_size` | `0.2`, `0.2` | Fractions of the whole dataset |

Search-space entries are `{"type": "int", "low", "high", "step"?}`,
`{"type": "float", "low", "high", "step"?, "log"?}` or
`{"type": "categorical", "choices": [...]}`.

`run` returns a `FineTuneResult` with `seed`, `best_params`,
`validation_score`, `test_score`, `split_sizes`, the fitted `model` and the
Optuna `study` (`None` when no search space was given).

## How tuning works

`FineTuner.run` follows the holdout-validation recipe from *Hands-On Machine
Learning*: the data is split into train / validation / test (stratified for
classification), every Optuna trial fits on the training split and is scored
on the validation split, the best hyperparameters are refit on train +
validation, and that final model is scored once on the test split.

Two things to keep in mind:

- `validation_score` is the best of `n_trials` scores on a single validation
  split, so it is optimistically biased; `test_score` is the honest estimate.
- Optuna's TPE sampler starts with `n_startup_trials` random trials (default
  10). With the default `n_trials=10` the study is therefore plain random
  search. Raise `n_trials` or set `n_startup_trials` lower to get model-based
  suggestions.

To evaluate the final model with any other metric or plot, rebuild the exact
test rows with `split_dataset(X, y, test_size=..., validation_size=...,
random_state=result.seed, stratify=y)` (drop `stratify` for regression).

## Tutorials

Six executed Jupyter notebooks in [`tutorials/`](tutorials/) take you from a
first run to a trustworthy evaluation. Start with
[`01_quickstart`](tutorials/01_quickstart.ipynb); the
[index](tutorials/README.md) describes the rest:

1. Quickstart: the holdout recipe, config, result, using the fitted model
2. Search spaces: spec types, the Optuna study, startup trials, failing trials
3. Classification: metrics for imbalanced data, confusion matrices on the test split, multiclass, string labels
4. Regression: RMSE, baselines, confidence intervals, pandas inputs, multi-output
5. Models and XGBoost: comparing model families, XGBoost, meta-estimators, edge cases
6. Reproducibility and pitfalls: seeds, validation optimism, split noise, saving models

## Limitations

- No preprocessing pipelines: `model_name` refers to a single estimator, so
  scale-sensitive models run on raw features.
- A single validation split, not cross-validation. Small datasets give noisy
  estimates; tutorial 06 shows how to measure that noise.
- `XGBClassifier` needs integer-encoded class labels; scikit-learn models
  accept strings directly.

## Development

```bash
pip install -e .[test]
pytest
```

The test suite covers the split, the workflow with and without a search
space, XGBoost seeding, reproducibility, config validation, model/problem
type checks, failing-trial handling, sparse inputs and the FIPS seed fallback.
The XGBoost test is skipped when `xgboost` cannot be loaded.

### Releasing to PyPI

Bump `version` in `pyproject.toml`, then build and upload with a PyPI API
token (`pip install build twine` once):

```bash
python -m build                      # writes dist/mloptune-<version>.tar.gz and .whl
python -m twine check dist/*
python -m twine upload dist/*        # username: __token__, password: the API token
```

Upload to TestPyPI first with `--repository testpypi` if you want to rehearse.

## Platform support

The framework is pure Python; everything platform-specific lives in its
dependencies, which all publish binary wheels for the platforms below.

| Platform | Status |
|---|---|
| macOS Apple Silicon (arm64) | Test suite run natively |
| macOS Intel (x86_64) | Test suite run under Rosetta 2 |
| Linux x86_64 and aarch64 | Test suite run in `python:3.12-slim` containers |
| Windows x86_64 and ARM64 | Wheels resolve for all dependencies; not executed |

Python 3.10 through 3.14 resolve on every platform except Windows ARM64,
which needs Python 3.12 or newer (older interpreters lack ARM64 wheels for
`xgboost`, `scikit-learn` or `pyyaml`). Seeds, chosen hyperparameters and
scores were identical across macOS arm64, Linux arm64 and Linux x86_64 for
the quick-start example. Notes:

- macOS needs `libomp` for XGBoost (see above); Windows needs the MSVC
  runtime, which XGBoost's wheels expect to be present.
- The experiment seed is derived with MD5. On FIPS-mode Linux, where the
  `seedhash` package's MD5 call is rejected, the framework falls back to the
  same derivation with `usedforsecurity=False`, so seeds stay identical.

## License

MIT. See [LICENSE](LICENSE).
