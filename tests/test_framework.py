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


# --- regression tests added while debugging -------------------------------------------

import pytest
from optuna.trial import TrialState

from mlfactory import framework
from mlfactory.framework import _supports_parameter


class _KwargsOnlyEstimator:
    """Mimics XGBoost's ``__init__(self, *, objective=..., **kwargs)`` signature."""

    def __init__(self, *, objective="reg:squarederror", **kwargs):
        self.objective = objective
        self.kwargs = kwargs

    def get_params(self, deep=True):
        return {"objective": self.objective, "random_state": None}


def test_supports_parameter_sees_through_kwargs_signatures():
    assert _supports_parameter(_KwargsOnlyEstimator, "random_state")
    assert not _supports_parameter(_KwargsOnlyEstimator, "does_not_exist")


def test_xgboost_models_receive_experiment_seed():
    if framework.XGBRegressor is None:
        pytest.skip("xgboost (or its OpenMP runtime) is not available")
    dataset = load_diabetes()
    config = FineTuneConfig(
        model_name="XGBRegressor",
        problem_type="regression",
        experiment_name="xgb-seed",
        model_kwargs={"n_estimators": 5},
    )

    result = FineTuner(config).run(dataset.data, dataset.target)

    assert result.model.get_params()["random_state"] == result.seed


def test_runs_are_reproducible_for_the_same_experiment_name():
    dataset = load_iris()
    config = FineTuneConfig(
        model_name="RandomForestClassifier",
        problem_type="classification",
        experiment_name="repro",
        n_trials=3,
        search_space={"n_estimators": {"type": "int", "low": 5, "high": 30}},
    )

    first = FineTuner(config).run(dataset.data, dataset.target)
    second = FineTuner(config).run(dataset.data, dataset.target)

    assert first.best_params == second.best_params
    assert first.validation_score == second.validation_score
    assert first.test_score == second.test_score


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"problem_type": "clustering"}, "problem_type"),
        ({"test_size": 0.6, "validation_size": 0.5}, "less than 1"),
        ({"test_size": 0.0}, "test_size"),
        ({"n_trials": 0, "search_space": {"a": {"type": "int", "low": 1, "high": 2}}}, "n_trials"),
        ({"search_space": {"a": {"type": "int", "min": 1, "max": 2}}}, "'low' and 'high'"),
        ({"search_space": {"a": {"type": "int", "low": 5, "high": 2}}}, "'low' must not exceed"),
        ({"search_space": {"a": {"type": "float", "low": 0.1, "high": 1, "log": True, "step": 0.1}}}, "step"),
        ({"search_space": {"a": {"type": "categorical", "choices": []}}}, "choices"),
        ({"search_space": {"a": {"type": "uniform", "low": 0, "high": 1}}}, "unsupported type"),
    ],
)
def test_config_rejects_invalid_settings_eagerly(kwargs, message):
    base = {"model_name": "RandomForestClassifier", "problem_type": "classification", "experiment_name": "x"}
    with pytest.raises(ValueError, match=message):
        FineTuneConfig(**{**base, **kwargs})


def test_model_must_match_problem_type():
    with pytest.raises(ValueError, match="is a regressor"):
        FineTuner(FineTuneConfig("RandomForestRegressor", "classification", "x"))
    with pytest.raises(ValueError, match="is a classifier"):
        FineTuner(FineTuneConfig("LogisticRegression", "regression", "x"))
    with pytest.raises(ValueError, match="Unknown model_name"):
        FineTuner(FineTuneConfig("KMeans", "classification", "x"))
    with pytest.raises(ValueError, match="Unknown model_name"):
        FineTuner(FineTuneConfig("PCA", "regression", "x"))


def test_failing_trials_do_not_abort_the_study():
    dataset = load_iris()
    config = FineTuneConfig(
        model_name="SVC",
        problem_type="classification",
        experiment_name="failing-trials",
        n_trials=6,
        search_space={"kernel": {"type": "categorical", "choices": ["linear", "bogus"]}},
    )

    result = FineTuner(config).run(dataset.data, dataset.target)

    states = {t.state for t in result.study.trials}
    assert TrialState.FAIL in states and TrialState.COMPLETE in states
    assert result.best_params == {"kernel": "linear"}


def test_all_trials_failing_raises_a_clear_error():
    dataset = load_iris()
    config = FineTuneConfig(
        model_name="SVC",
        problem_type="classification",
        experiment_name="all-fail",
        n_trials=2,
        search_space={"kernel": {"type": "categorical", "choices": ["bogus"]}},
    )

    with pytest.raises(RuntimeError, match="All 2 Optuna trials failed"):
        FineTuner(config).run(dataset.data, dataset.target)


def test_result_exposes_study_and_startup_trials_are_configurable():
    dataset = load_iris()
    config = FineTuneConfig(
        model_name="DecisionTreeClassifier",
        problem_type="classification",
        experiment_name="study",
        n_trials=4,
        n_startup_trials=2,
        search_space={"max_depth": {"type": "int", "low": 1, "high": 6}},
    )

    result = FineTuner(config).run(dataset.data, dataset.target)

    assert result.study is not None
    assert len(result.study.trials) == 4
    assert result.study.sampler._n_startup_trials == 2


def test_sparse_inputs_survive_the_final_refit():
    import scipy.sparse as sp

    dataset = load_iris()
    config = FineTuneConfig(
        model_name="LogisticRegression",
        problem_type="classification",
        experiment_name="sparse",
        n_trials=2,
        search_space={"C": {"type": "float", "low": 0.1, "high": 10.0}},
        model_kwargs={"max_iter": 500},
    )

    result = FineTuner(config).run(sp.csr_matrix(dataset.data), dataset.target)

    assert 0.0 <= result.test_score <= 1.0
    assert result.model.n_features_in_ == 4


def test_split_dataset_returns_train_validation_partition_consistently():
    X = np.arange(50).reshape(25, 2)
    y = np.array([0, 1, 0, 1, 0] * 5)

    split = split_dataset(X, y, test_size=0.2, validation_size=0.2, random_state=7, stratify=y)

    merged = np.concatenate([split["X_train"], split["X_validation"]])
    assert sorted(merged.tolist()) == sorted(split["X_train_validation"].tolist())
    assert len(split["y_train_validation"]) == 20


def test_unknown_hyperparameter_names_are_rejected_before_training():
    with pytest.raises(ValueError, match=r"no hyperparameter\(s\) \['n_estimator'\]"):
        FineTuner(
            FineTuneConfig(
                "RandomForestClassifier",
                "classification",
                "x",
                search_space={"n_estimator": {"type": "int", "low": 5, "high": 10}},
            )
        )
    with pytest.raises(ValueError, match="max_dept"):
        FineTuner(FineTuneConfig("RandomForestClassifier", "classification", "x", model_kwargs={"max_dept": 3}))


def test_config_snapshots_caller_mappings():
    space = {"max_depth": {"type": "int", "low": 1, "high": 5}}
    kwargs = {"criterion": "gini"}
    config = FineTuneConfig("DecisionTreeClassifier", "classification", "x", search_space=space, model_kwargs=kwargs)

    space["max_depth"]["low"] = 99
    kwargs["criterion"] = "bogus"

    assert config.search_space["max_depth"]["low"] == 1
    assert config.model_kwargs["criterion"] == "gini"


def test_seed_fallback_matches_seedhash(monkeypatch):
    import hashlib

    from mlfactory import framework

    expected = framework._build_seed("iris-workflow")

    class _FipsRejects:
        def __init__(self, name):
            raise ValueError("[digital envelope routines] unsupported")

    monkeypatch.setattr(framework, "SeedHashGenerator", _FipsRejects)
    assert framework._build_seed("iris-workflow") == expected
    assert expected == int(hashlib.md5(b"iris-workflow").hexdigest(), 16) % (2**32)
    with pytest.raises(ValueError):
        framework._build_seed("")
