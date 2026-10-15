"""Phase 3: hyperparameter search (Optuna) + decision-threshold tuning.

All search ranges and toggles come from params.yaml (see
amr_guard.utils.config.load_params) — nothing here is hardcoded.

optuna is an optional dependency (declared in pyproject.toml but not
required to be installed). If it isn't installed, tune_xgboost() falls
back to the static xgboost params from params.yaml with a printed
warning, so the pipeline still runs end-to-end without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from amr_guard.utils.config import load_params

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False


def _suggest_param(trial, name: str, spec: dict):
    if "log" in spec:
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec["log"])
    if "step" in spec and isinstance(spec["step"], int):
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec["step"])
    if isinstance(spec["low"], int) and isinstance(spec["high"], int) and "step" not in spec:
        return trial.suggest_int(name, spec["low"], spec["high"])
    return trial.suggest_float(name, spec["low"], spec["high"])


def _cv_score(params: dict, X: pd.DataFrame, y: pd.Series, cv_folds: int, metric: str, random_state: int) -> float:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        scale_pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        clf = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            use_label_encoder=False,
        )
        clf.fit(X_tr, y_tr, verbose=False)
        preds = clf.predict(X_val)
        if metric == "f1":
            scores.append(f1_score(y_val, preds, zero_division=0))
        else:
            raise ValueError(f"Unsupported optuna metric: {metric}")
    return float(np.mean(scores))


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, params: dict | None = None, random_state: int | None = None) -> dict:
    """Returns a dict of tuned XGBoost hyperparameters. Falls back to the
    static params.yaml xgboost block if optuna is unavailable or disabled.
    """
    config = load_params()
    optuna_config = params or config["optuna"]
    random_state = random_state if random_state is not None else config["data"]["random_state"]
    static_params = config["xgboost"]

    if not OPTUNA_AVAILABLE or not optuna_config.get("enabled", True):
        print("[optuna] not installed or disabled, falling back to static xgboost params.")
        return {
            "n_estimators": static_params["n_estimators"],
            "max_depth": static_params["max_depth"],
            "learning_rate": static_params["learning_rate"],
            "subsample": static_params["subsample"],
            "colsample_bytree": static_params["colsample_bytree"],
            "eval_metric": static_params["eval_metric"],
        }

    search_space = optuna_config["search_space"]
    eval_metric = static_params["eval_metric"]
    cv_folds = optuna_config["cv_folds"]
    metric = optuna_config["metric"]

    def objective(trial):
        trial_params = {name: _suggest_param(trial, name, spec) for name, spec in search_space.items()}
        trial_params["eval_metric"] = eval_metric
        return _cv_score(trial_params, X_train, y_train, cv_folds, metric, random_state)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction=optuna_config["direction"], sampler=sampler)
    study.optimize(objective, n_trials=optuna_config["n_trials"], timeout=optuna_config.get("timeout_seconds"))

    best_params = dict(study.best_params)
    best_params["eval_metric"] = eval_metric
    print(f"[optuna] best {metric}={study.best_value:.4f} params={best_params}")
    return best_params


def tune_decision_threshold(model, X_val: pd.DataFrame, y_val: pd.Series, params: dict | None = None) -> float:
    """Scans candidate thresholds from params.yaml and returns the one that
    maximizes the configured metric on the validation set. Falls back to
    0.5 if threshold tuning is disabled.
    """
    config = params or load_params()["threshold_tuning"]
    if not config.get("enabled", True):
        return 0.5

    spec = config["candidate_thresholds"]
    thresholds = np.arange(spec["low"], spec["high"] + spec["step"] / 2, spec["step"])
    probs = model.predict_proba(X_val)[:, 1]

    best_threshold, best_score = 0.5, -1.0
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        if config["metric"] == "f1":
            score = f1_score(y_val, preds, zero_division=0)
        else:
            raise ValueError(f"Unsupported threshold_tuning metric: {config['metric']}")
        if score > best_score:
            best_threshold, best_score = float(threshold), score

    print(f"[threshold_tuning] best threshold={best_threshold:.2f} {config['metric']}={best_score:.4f}")
    return best_threshold
