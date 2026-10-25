"""Compare taxi-duration regressors on one shared train/validation split."""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)


def rmsle(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.clip(np.asarray(predicted), 0, None)
    return np.sqrt(np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2))


def candidate_models(random_state=42):
    return {
        "Histogram Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=120, max_leaf_nodes=31, learning_rate=0.08,
            random_state=random_state,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=60, max_depth=20, min_samples_leaf=4,
            max_features=0.8, n_jobs=-1, random_state=random_state,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=60, max_depth=20, min_samples_leaf=4,
            max_features=0.8, n_jobs=-1, random_state=random_state,
        ),
    }


def compare_models(X, y, train_idx, valid_idx, sample_size=150_000, random_state=42):
    """Return validation scores and fitted candidates trained on identical rows."""
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(train_idx, size=min(sample_size, len(train_idx)), replace=False)
    target = np.log1p(y[sample_idx])
    baseline_seconds = np.expm1(target.mean())
    rows = [{
        "model": "Constant baseline",
        "train_rmsle": rmsle(y[sample_idx], np.full(len(sample_idx), baseline_seconds)),
        "valid_rmsle": rmsle(y[valid_idx], np.full(len(valid_idx), baseline_seconds)),
    }]
    fitted = {}

    for name, model in candidate_models(random_state).items():
        model.fit(X.iloc[sample_idx], target)
        train_pred = np.expm1(model.predict(X.iloc[sample_idx])).clip(min=0)
        valid_pred = np.expm1(model.predict(X.iloc[valid_idx])).clip(min=0)
        rows.append({
            "model": name,
            "train_rmsle": rmsle(y[sample_idx], train_pred),
            "valid_rmsle": rmsle(y[valid_idx], valid_pred),
        })
        fitted[name] = model

    return pd.DataFrame(rows), fitted
