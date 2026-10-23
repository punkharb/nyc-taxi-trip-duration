# NYC Taxi Trip Duration — first competition baseline

This is a small, runnable first pass for the [Kaggle NYC Taxi Trip Duration competition](https://www.kaggle.com/competitions/nyc-taxi-trip-duration). The goal is to predict each trip's duration in seconds. The 2017 competition is closed, but Kaggle currently labels its practice uploads **Late Submission**.

## What I did

1. Checked the real train and test files: 1,458,644 labeled trips and 625,134 test trips. Both cover January–June 2016, and the train file has no missing values.
2. Explicitly built features from pickup time, pickup and destination coordinates, passenger count, vendor, and store-and-forward flag. Straight-line distance is derived from the coordinates.
3. Excluded `dropoff_datetime` because it is known only when the ride ends and directly reveals the target.
4. Used a fixed random 80/20 holdout. Unlike future sales forecasting, this competition's test trips are interleaved across the same calendar period as train. A time holdout would answer a different question.
5. Compared a constant baseline with a histogram gradient boosting regressor, both using Kaggle's RMSLE metric. The model uses a 300,000-trip training sample for a fast first experiment.
6. Refit on a 500,000-trip sample and created `id,trip_duration` predictions for every test row.

| Local validation | RMSLE (lower is better) |
|---|---:|
| Constant baseline | 0.79661 |
| Gradient boosting | 0.42871 |

The model's training-sample RMSLE was 0.42202, close to validation's 0.42871. This suggests limited overfitting on this split, not guaranteed performance on future trips. The notebook checks output row count, ID order, finite values, and nonnegative predictions.

These scores were recorded from the local run. Notebook outputs are cleared in Git to keep the repository compact; rerunning the notebook reproduces the local validation comparison.

A **late practice submission** on Kaggle scored **0.43193 public / 0.42971 private** (23 October 2026). This is separate from the original 2017 contest result.

## How this differs from the previous projects

- **Store Sales:** forecasting future dates for many store/product series; must validate past → future. Here each row is one taxi ride and train/test dates overlap, so this is tabular regression with time-derived features.
- **Telco Churn:** predicts a yes/no outcome. Here the target is a continuous number of seconds, so the metric and error interpretation differ.
- **Online Retail RFM:** summarizes and segments customers with rules. Here we train a supervised model against known durations.

## Limits

- The random holdout resembles this competition split, but it does not prove performance on future months or other cities.
- Destination coordinates are provided by Kaggle. In a real product, they would be available only after the rider enters a destination.
- A few abnormal coordinates and extremely long trips remain. This first pass does not claim to solve all data-quality problems.
- Late submissions are practice feedback, not results from the original 2017 contest.

## Run

Attach the competition data to a Kaggle notebook and run `nyc_taxi_trip_duration.ipynb` top to bottom. Locally, put `train.csv` and `test.csv` under `data/` (or their single-CSV ZIP files) and run from this folder. The notebook saves `output/submission.csv` locally or `/kaggle/working/submission.csv` on Kaggle. Raw data and predictions are ignored by Git.

For a local run, use Python 3.10+ and install `numpy`, `pandas`, `scikit-learn`, and `jupyter`. Install `pytest` too if you want to run `python -m pytest tests -q`. Kaggle notebooks already provide the modeling libraries.
