from __future__ import annotations

import copy
import hashlib
import inspect
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping

import optuna
from optuna.trial import TrialState
from seedhash import SeedHashGenerator
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # noqa: BLE001
    # xgboost raises XGBoostError (a ValueError subclass, not ImportError) when its
    # native library cannot be loaded, e.g. on macOS without libomp. Either way the
    # rest of the framework must stay importable.
    XGBClassifier = None
    XGBRegressor = None

PROBLEM_TYPES = ("classification", "regression")
_TYPE_FILTERS = {"classification": "classifier", "regression": "regressor"}
_SEARCH_SPACE_TYPES = ("int", "float", "categorical")


@lru_cache(maxsize=len(PROBLEM_TYPES))
def _registry(problem_type: str) -> dict[str, type]:
    """Estimators that are valid for ``problem_type`` (classifiers or regressors only)."""
    registry = dict(all_estimators(type_filter=_TYPE_FILTERS[problem_type]))
    if problem_type == "classification" and XGBClassifier is not None:
        registry["XGBClassifier"] = XGBClassifier
    if problem_type == "regression" and XGBRegressor is not None:
        registry["XGBRegressor"] = XGBRegressor
    return registry


def _supports_parameter(estimator_cls: type, parameter_name: str) -> bool:
    parameters = inspect.signature(estimator_cls.__init__).parameters
    if parameter_name in parameters:
        return True
    # Estimators such as XGBClassifier declare ``**kwargs`` and only reveal their real
    # parameters through ``get_params()``.
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return False
    try:
        return parameter_name in estimator_cls().get_params()
    except Exception:  # noqa: BLE001 - estimator cannot be built with defaults.
        return False


def _known_parameters(estimator_cls: type) -> set[str] | None:
    """Names the estimator's constructor accepts, or ``None`` if they cannot be determined."""
    parameters = inspect.signature(estimator_cls.__init__).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return {name for name in parameters if name != "self"}
    try:
        return set(estimator_cls().get_params())
    except Exception:  # noqa: BLE001 - estimator cannot be built with defaults.
        return None


def _build_seed(experiment_name: str) -> int:
    try:
        seed_number = SeedHashGenerator(experiment_name).seed_number
    except ValueError as exc:
        if not experiment_name:
            raise
        # seedhash calls hashlib.md5() without usedforsecurity=False, which FIPS-mode
        # Linux rejects. Reproduce its exact derivation with the non-security flag.
        digest = hashlib.md5(experiment_name.encode("utf-8"), usedforsecurity=False).hexdigest()  # noqa: S324
        seed_number = int(digest, 16)
        del exc
    return seed_number % (2**32)


def _validate_search_space(search_space: Mapping[str, Mapping[str, Any]]) -> None:
    for name, spec in search_space.items():
        if not isinstance(spec, Mapping) or "type" not in spec:
            raise ValueError(f"search_space[{name!r}] must be a mapping with a 'type' key.")
        spec_type = spec["type"]
        if spec_type not in _SEARCH_SPACE_TYPES:
            raise ValueError(
                f"search_space[{name!r}] has unsupported type {spec_type!r}; "
                f"expected one of {_SEARCH_SPACE_TYPES}."
            )
        if spec_type == "categorical":
            choices = spec.get("choices")
            if not choices:
                raise ValueError(f"search_space[{name!r}] of type 'categorical' needs a non-empty 'choices' list.")
            continue
        if "low" not in spec or "high" not in spec:
            raise ValueError(f"search_space[{name!r}] of type {spec_type!r} needs 'low' and 'high' keys.")
        if spec["low"] > spec["high"]:
            raise ValueError(f"search_space[{name!r}]: 'low' must not exceed 'high'.")
        if spec_type == "float" and spec.get("log", False) and spec.get("step") is not None:
            raise ValueError(f"search_space[{name!r}]: 'step' cannot be combined with log=True.")


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
        # Kept as sliced by scikit-learn so the final refit works for any container it
        # supports (NumPy, pandas, scipy sparse, lists) without re-concatenating.
        "X_train_validation": X_train_val,
        "y_train_validation": y_train_val,
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
    n_startup_trials: int | None = None
    """Random trials before Optuna's TPE sampler starts modelling the objective.

    ``None`` keeps Optuna's default (10). Note that with the default ``n_trials=10`` the
    whole study is therefore random search; lower this (or raise ``n_trials``) to get
    model-based suggestions.
    """

    def __post_init__(self) -> None:
        if self.problem_type not in PROBLEM_TYPES:
            raise ValueError(f"problem_type must be one of {PROBLEM_TYPES}.")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")
        if self.test_size + self.validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1.")
        if self.search_space and self.n_trials < 1:
            raise ValueError("n_trials must be at least 1 when a search_space is given.")
        if self.n_startup_trials is not None and self.n_startup_trials < 0:
            raise ValueError("n_startup_trials must be non-negative.")
        # Snapshot the mappings: the dataclass is frozen, but the caller's dicts are not.
        object.__setattr__(self, "search_space", copy.deepcopy(dict(self.search_space)))
        object.__setattr__(self, "model_kwargs", copy.deepcopy(dict(self.model_kwargs)))
        _validate_search_space(self.search_space)


@dataclass(frozen=True)
class FineTuneResult:
    seed: int
    best_params: dict[str, Any]
    validation_score: float
    test_score: float
    split_sizes: dict[str, int]
    model: Any
    study: optuna.Study | None = None
    """The Optuna study (``None`` when no search_space was given). Inspect
    ``study.trials_dataframe()`` to see every configuration that was evaluated."""


class FineTuner:
    def __init__(self, config: FineTuneConfig):
        self.config = config
        self._registry = _registry(config.problem_type)
        if config.model_name not in self._registry:
            other = [p for p in PROBLEM_TYPES if p != config.problem_type][0]
            if config.model_name in _registry(other):
                raise ValueError(
                    f"{config.model_name} is a {_TYPE_FILTERS[other]}, but problem_type is "
                    f"{config.problem_type!r}."
                )
            raise ValueError(
                f"Unknown model_name: {config.model_name}. It must be a scikit-learn "
                f"{_TYPE_FILTERS[config.problem_type]} (see sklearn.utils.all_estimators) "
                "or an XGBoost alias."
            )
        known = _known_parameters(self._registry[config.model_name])
        if known is not None:
            unknown = sorted((set(config.model_kwargs) | set(config.search_space)) - known)
            if unknown:
                raise ValueError(
                    f"{config.model_name} has no hyperparameter(s) {unknown}; "
                    f"valid names include {sorted(known)[:12]}..."
                )

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
        study: optuna.Study | None = None
        if self.config.search_space:
            sampler_kwargs: dict[str, Any] = {"seed": seed}
            if self.config.n_startup_trials is not None:
                sampler_kwargs["n_startup_trials"] = self.config.n_startup_trials
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(**sampler_kwargs),
            )
            failures: list[BaseException] = []

            def objective(trial: optuna.Trial) -> float:
                try:
                    return self._objective(
                        trial,
                        estimator_cls,
                        base_params,
                        scorer,
                        split["X_train"],
                        split["y_train"],
                        split["X_validation"],
                        split["y_validation"],
                    )
                except Exception as exc:  # noqa: BLE001 - recorded, then handed to Optuna.
                    failures.append(exc)
                    raise

            # A single bad hyper-parameter combination must not abort the whole study;
            # Optuna marks such trials as FAIL and keeps going.
            study.optimize(objective, n_trials=self.config.n_trials, catch=(Exception,))
            completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
            if not completed:
                raise RuntimeError(
                    f"All {len(study.trials)} Optuna trials failed; last error: {failures[-1]!r}"
                ) from failures[-1]
            best_params = dict(study.best_params)
            validation_score = float(study.best_value)
        else:
            candidate = estimator_cls(**base_params)
            candidate.fit(split["X_train"], split["y_train"])
            validation_score = float(scorer(candidate, split["X_validation"], split["y_validation"]))

        final_params = {**base_params, **best_params}
        final_model = estimator_cls(**final_params)
        final_model.fit(split["X_train_validation"], split["y_train_validation"])
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
            study=study,
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
