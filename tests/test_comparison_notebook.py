import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / "02_model_comparison.ipynb"


def test_model_comparison_is_a_separate_runnable_notebook():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        compile("".join(cell["source"]), str(NOTEBOOK), "exec")


def test_notebook_creates_competition_submission(tmp_path, monkeypatch):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    def trips(count):
        rows = np.arange(count)
        return pd.DataFrame({
            "id": [f"id{i}" for i in rows],
            "vendor_id": np.ones(count, dtype=int),
            "pickup_datetime": ["2016-01-04 08:30:00"] * count,
            "passenger_count": np.ones(count, dtype=int),
            "pickup_longitude": -73.98 + rows * 0.0001,
            "pickup_latitude": np.full(count, 40.75),
            "dropoff_longitude": -73.96 + rows * 0.0001,
            "dropoff_latitude": np.full(count, 40.76),
            "store_and_fwd_flag": ["N"] * count,
        })

    train = trips(30)
    train["trip_duration"] = 300 + np.arange(30) * 10
    train.to_csv(data_dir / "train.csv", index=False)
    test = trips(3)
    test.to_csv(data_dir / "test.csv", index=False)

    scope = {"display": lambda _: None}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            if "eda-plot" in cell.get("metadata", {}).get("tags", []) and importlib.util.find_spec("matplotlib") is None:
                continue
            exec("".join(cell["source"]), scope)

    submission = pd.read_csv(tmp_path / "output" / "submission.csv")
    assert submission.columns.tolist() == ["id", "trip_duration"]
    assert submission["id"].tolist() == test["id"].tolist()
    assert submission["trip_duration"].notna().all()
    assert submission["trip_duration"].ge(0).all()


def test_notebook_finds_kaggle_competition_mount(tmp_path):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "data-locator" in cell.get("metadata", {}).get("tags", [])
    )
    scope = {}
    exec(source, scope)
    mounted_dir = tmp_path / "input" / "competitions" / "nyc-taxi-trip-duration"
    mounted_dir.mkdir(parents=True)
    (mounted_dir / "train.zip").touch()
    (mounted_dir / "test.zip").touch()

    assert scope["locate_data_dir"](tmp_path / "input", tmp_path / "local") == mounted_dir


def test_notebook_eda_reports_data_quality_before_modeling():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    train = pd.DataFrame({
        "id": ["a", "b", "c", "c"],
        "pickup_datetime": ["2016-01-01 08:00:00"] * 4,
        "passenger_count": [1, 0, 2, 2],
        "pickup_latitude": [40.75] * 4,
        "pickup_longitude": [-73.98] * 4,
        "dropoff_latitude": [40.76] * 4,
        "dropoff_longitude": [-73.96] * 4,
        "trip_duration": [300, 0, 600, 600],
    })
    eda_cell = next(
        cell for cell in notebook["cells"]
        if "eda-quality" in cell.get("metadata", {}).get("tags", [])
    )
    scope = {"pd": pd, "np": np, "train": train, "display": lambda _: None}
    exec("".join(eda_cell["source"]), scope)
    checks = scope["quality_checks"]
    assert checks["rows"] == 4
    assert checks["exact_duplicate_rows"] == 1
    assert checks["duplicate_trip_ids"] == 1
    assert checks["nonpositive_duration"] == 1
    assert checks["nonpositive_passenger_count"] == 1
