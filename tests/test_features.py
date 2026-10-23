import json
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / "nyc_taxi_trip_duration.ipynb"


def test_features_use_only_prediction_time_information():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "feature-engineering" in cell.get("metadata", {}).get("tags", [])
    )
    scope = {"np": np, "pd": pd}
    exec(source, scope)

    trips = pd.DataFrame(
        {
            "pickup_datetime": ["2016-01-04 08:30:00"],
            "dropoff_datetime": ["2016-01-04 08:40:00"],
            "trip_duration": [600],
            "vendor_id": [1],
            "passenger_count": [2],
            "pickup_longitude": [-73.98],
            "pickup_latitude": [40.75],
            "dropoff_longitude": [-73.98],
            "dropoff_latitude": [40.75],
            "store_and_fwd_flag": ["N"],
        }
    )

    features = scope["build_features"](trips)

    assert "dropoff_datetime" not in features
    assert "trip_duration" not in features
    assert features.loc[0, "distance_km"] == 0
    assert features.loc[0, "pickup_hour"] == 8
    assert features.loc[0, "pickup_weekday"] == 0
    assert features.notna().all().all()


def test_rmsle_matches_a_worked_example():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "evaluation" in cell.get("metadata", {}).get("tags", [])
    )
    scope = {"np": np}
    exec(source, scope)

    assert np.isclose(scope["rmsle"]([0, 3], [0, 1]), np.log(2) / np.sqrt(2))
