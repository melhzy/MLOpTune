# MLFactory

MLFactory is a lightweight machine learning fine-tuning framework for
scikit-learn-compatible models.

## Features

- Deterministic experiment seeding powered by `seedhash`
- Train/validation/test splitting with stratification for classification tasks
- Optuna-based hyperparameter tuning against a validation split
- Support for every estimator exposed by `sklearn.utils.all_estimators()`
- Built-in `xgboost` aliases for XGBoost classifiers and regressors

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
```

## Development

Install the project and test dependencies:

```bash
pip install -e .[test]
pytest
```
