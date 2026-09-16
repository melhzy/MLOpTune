import numpy as np
from sklearn.datasets import load_diabetes, load_iris

from mlfactory import FineTuneConfig, FineTuner, split_dataset
from mlfactory.framework import _build_seed


def test_split_dataset_creates_train_validation_test_partitions():
    X = np.arange(50).reshape(25, 2)
    y = np.array([0, 1, 0, 1, 0] * 5)

    split = split_dataset(
        X,
        y,
        test_size=0.2,
        validation_size=0.2,
        random_state=7,
        stratify=y,
    )

    assert split["X_train"].shape[0] == 15
    assert split["X_validation"].shape[0] == 5
    assert split["X_test"].shape[0] == 5


def test_fine_tuner_runs_classification_workflow_with_optuna():
    dataset = load_iris()
    config = FineTuneConfig(
        model_name="RandomForestClassifier",
        problem_type="classification",
        experiment_name="iris-workflow",
        n_trials=2,
        search_space={
            "n_estimators": {"type": "int", "low": 5, "high": 10},
            "max_depth": {"type": "int", "low": 2, "high": 4},
        },
    )

    result = FineTuner(config).run(dataset.data, dataset.target)

    assert result.seed == _build_seed("iris-workflow")
    assert set(result.best_params) == {"n_estimators", "max_depth"}
    assert result.split_sizes == {"train": 90, "validation": 30, "test": 30}
    assert 0.0 <= result.validation_score <= 1.0
    assert 0.0 <= result.test_score <= 1.0


def test_fine_tuner_runs_regression_workflow_without_search_space():
    dataset = load_diabetes()
    config = FineTuneConfig(
        model_name="RandomForestRegressor",
        problem_type="regression",
        experiment_name="diabetes-workflow",
        model_kwargs={"n_estimators": 10},
    )

    result = FineTuner(config).run(dataset.data, dataset.target)

    assert result.best_params == {}
    assert result.split_sizes == {"train": 264, "validation": 89, "test": 89}
    assert np.isfinite(result.validation_score)
    assert np.isfinite(result.test_score)
    assert result.validation_score <= 0.0
    assert result.test_score <= 0.0
