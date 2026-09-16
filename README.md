# MLFactory

MLFactory is a lightweight machine learning fine-tuning framework for
scikit-learn-compatible models.

## Features

- Deterministic experiment seeding powered by `seedhash`
- Train/validation/test splitting with stratification for classification tasks
- Optuna-based hyperparameter tuning against a validation split
- Support for every classifier or regressor exposed by `sklearn.utils.all_estimators()`
  (the model must match `problem_type`; mismatches are rejected up front)
- Built-in `xgboost` aliases for XGBoost classifiers and regressors (optional: the
  framework still imports if `xgboost` or its OpenMP runtime is missing)
- Eager validation of the config and search space, and tolerance of individual
  failing Optuna trials

## Quick start

```python
from sklearn.datasets import load_iris

from mlfactory import FineTuneConfig, FineTuner

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

## How tuning works

`FineTuner.run` follows the holdout-validation recipe: the data is split into
train / validation / test (stratified for classification), every Optuna trial
fits on the training split and is scored on the validation split, the best
hyperparameters are refit on train + validation, and that final model is scored
once on the untouched test split.

Two things to keep in mind:

- `validation_score` is the best of `n_trials` scores on a single validation
  split, so it is optimistically biased; `test_score` is the honest estimate.
- Optuna's TPE sampler starts with `n_startup_trials` random trials (default 10).
  With the default `n_trials=10` the study is therefore plain random search. Raise
  `n_trials` or set `n_startup_trials` lower to get model-based suggestions.

Search-space entries take the form `{"type": "int"|"float"|"categorical", ...}`
with `low`/`high` (plus optional `step`, and `log` for floats) or `choices`.

## Development

Install the project and test dependencies:

```bash
pip install -e .[test]
pytest
```

On macOS, `xgboost` needs the OpenMP runtime (`brew install libomp`). Without
it the XGBoost aliases are simply unavailable and the XGBoost test is skipped.

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
`xgboost`, `scikit-learn` or `pyyaml`). Seeds, chosen
hyperparameters and scores were identical across macOS arm64, Linux arm64 and
Linux x86_64 for the quick-start example. Notes:

- macOS needs `libomp` for XGBoost (see above); Windows needs the MSVC
  runtime, which XGBoost's wheels expect to be present.
- The experiment seed is derived with MD5. On FIPS-mode Linux, where the
  `seedhash` package's MD5 call is rejected, the framework falls back to the
  same derivation with `usedforsecurity=False`, so seeds stay identical.
