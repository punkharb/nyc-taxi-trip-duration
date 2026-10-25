import numpy as np
import pandas as pd

from model_comparison import compare_models


def test_compare_models_scores_candidates_on_one_holdout():
    rng = np.random.default_rng(7)
    features = pd.DataFrame({
        "distance_km": rng.uniform(0, 15, 100),
        "pickup_hour": rng.integers(0, 24, 100),
    })
    seconds = 120 + 80 * features["distance_km"].to_numpy()
    train_idx = np.arange(80)
    valid_idx = np.arange(80, 100)

    scores, fitted = compare_models(
        features, seconds, train_idx, valid_idx, sample_size=60, random_state=42
    )

    assert scores["model"].tolist() == [
        "Constant baseline", "Histogram Gradient Boosting", "Random Forest", "Extra Trees"
    ]
    assert set(fitted) == set(scores["model"]) - {"Constant baseline"}
    assert scores["valid_rmsle"].notna().all()
    assert np.isfinite(scores["valid_rmsle"]).all()
    assert (scores["valid_rmsle"] >= 0).all()
