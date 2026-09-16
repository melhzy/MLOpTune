from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import optuna
from seedhash import SeedHashGenerator
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover - dependency is declared, but keep imports defensive.
    XGBClassifier = None
    XGBRegressor = None


def _registry() -> dict[str, type]:
    registry = {name: estimator for name, estimator in all_estimators()}
    if XGBClassifier is not None:
        registry["XGBClassifier"] = XGBClassifier
    if XGBRegressor is not None:
        registry["XGBRegressor"] = XGBRegressor
    return registry


def _supports_parameter(estimator_cls: type, parameter_name: str) -> bool:
    return parameter_name in inspect.signature(estimator_cls.__init__).parameters


def _build_seed(experiment_name: str) -> int:
    return SeedHashGenerator(experiment_name).seed_number % (2**32)


def split_dataset(
    X: Any,
    y: Any,
    *,
    test_size: float,
    validation_size: float,
    random_state: int,
    stratify: Any = None,
) -> dict[str, Any]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    validation_share = validation_size / (1 - test_size)
    train_val_stratify = y_train_val if stratify is not None else None
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_share,
        random_state=random_state,
        stratify=train_val_stratify,
    )

    return {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
    }


@dataclass(frozen=True)
class FineTuneConfig:
    model_name: str
    problem_type: str
    experiment_name: str
    test_size: float = 0.2
    validation_size: float = 0.2
    scoring: str | None = None
    n_trials: int = 10
    search_space: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FineTuneResult:
    seed: int
    best_params: dict[str, Any]
    validation_score: float
    test_score: float
    split_sizes: dict[str, int]
    model: Any


class FineTuner:
    def __init__(self, config: FineTuneConfig):
        self.config = config
        self._registry = _registry()
        if config.model_name not in self._registry:
            raise ValueError(f"Unknown model_name: {config.model_name}")
        if config.problem_type not in {"classification", "regression"}:
            raise ValueError("problem_type must be 'classification' or 'regression'.")

    def run(self, X: Any, y: Any) -> FineTuneResult:
        """Tune on the validation split, then refit on train+validation for final test scoring."""
        seed = _build_seed(self.config.experiment_name)
        scoring = self.config.scoring or self._default_scoring()
        scorer = get_scorer(scoring)
        stratify = y if self.config.problem_type == "classification" else None
        split = split_dataset(
            X,
            y,
            test_size=self.config.test_size,
            validation_size=self.config.validation_size,
            random_state=seed,
            stratify=stratify,
        )

        estimator_cls = self._registry[self.config.model_name]
        base_params = dict(self.config.model_kwargs)
        if "random_state" not in base_params and _supports_parameter(estimator_cls, "random_state"):
            base_params["random_state"] = seed

        best_params: dict[str, Any] = {}
        validation_score = float("-inf")
        if self.config.search_space:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=seed),
            )
            study.optimize(
                lambda trial: self._objective(
                    trial,
                    estimator_cls,
                    base_params,
                    scorer,
                    split["X_train"],
                    split["y_train"],
                    split["X_validation"],
                    split["y_validation"],
                ),
                n_trials=self.config.n_trials,
            )
            best_params = dict(study.best_params)
            validation_score = float(study.best_value)
        else:
            candidate = estimator_cls(**base_params)
            candidate.fit(split["X_train"], split["y_train"])
            validation_score = float(scorer(candidate, split["X_validation"], split["y_validation"]))

        final_model = estimator_cls(**base_params, **best_params)
        final_model.fit(
            self._concat(split["X_train"], split["X_validation"]),
            self._concat(split["y_train"], split["y_validation"]),
        )
        test_score = float(scorer(final_model, split["X_test"], split["y_test"]))

        return FineTuneResult(
            seed=seed,
            best_params=best_params,
            validation_score=validation_score,
            test_score=test_score,
            split_sizes={
                "train": len(split["y_train"]),
                "validation": len(split["y_validation"]),
                "test": len(split["y_test"]),
            },
            model=final_model,
        )

    def _objective(
        self,
        trial: optuna.Trial,
        estimator_cls: type,
        base_params: dict[str, Any],
        scorer: Any,
        X_train: Any,
        y_train: Any,
        X_validation: Any,
        y_validation: Any,
    ) -> float:
        candidate_params = dict(base_params)
        for name, spec in self.config.search_space.items():
            candidate_params[name] = self._suggest_parameter(trial, name, spec)

        candidate = estimator_cls(**candidate_params)
        candidate.fit(X_train, y_train)
        return float(scorer(candidate, X_validation, y_validation))

    def _default_scoring(self) -> str:
        if self.config.problem_type == "classification":
            return "accuracy"
        return "neg_mean_squared_error"

    @staticmethod
    def _concat(left: Any, right: Any) -> Any:
        if isinstance(left, np.ndarray):
            return np.concatenate([left, right])
        if hasattr(left, "iloc"):
            import pandas as pd

            return pd.concat([left, right])  # pragma: no cover
        return list(left) + list(right)

    @staticmethod
    def _suggest_parameter(trial: optuna.Trial, name: str, spec: Mapping[str, Any]) -> Any:
        spec_type = spec["type"]
        if spec_type == "int":
            return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        if spec_type == "float":
            return trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step"),
                log=spec.get("log", False),
            )
        if spec_type == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        raise ValueError(f"Unsupported search space type: {spec_type}")
