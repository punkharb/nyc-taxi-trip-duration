# NYC Taxi Trip Duration — regression model comparison

This is a small, runnable first pass for the [Kaggle NYC Taxi Trip Duration competition](https://www.kaggle.com/competitions/nyc-taxi-trip-duration). The goal is to predict each trip's duration in seconds. The 2017 competition is closed, but Kaggle currently labels its practice uploads **Late Submission**.

## Notebooks

- [02_model_comparison.ipynb](02_model_comparison.ipynb) — a separate, self-contained Kaggle-ready notebook comparing three models on the same validation set, then saving `/kaggle/working/submission.csv`. You can also [open it in Kaggle](https://www.kaggle.com/code/punkharb/nyc-taxi-trip-duration-model-comparison) to run and submit it from your account.
- [nyc_taxi_trip_duration.ipynb](nyc_taxi_trip_duration.ipynb) — the end-to-end pipeline that refits the selected model and creates the Kaggle submission. Keep `model_comparison.py` beside this notebook when running it.

## What I did

1. Checked the real train and test files: 1,458,644 labeled trips and 625,134 test trips. Both cover January–June 2016, and the train file has no missing values.
2. Explicitly built features from pickup time, pickup and destination coordinates, passenger count, vendor, and store-and-forward flag. Straight-line distance is derived from the coordinates.
3. Excluded `dropoff_datetime` because it is known only when the ride ends and directly reveals the target.
4. Used a fixed random 80/20 holdout. Unlike future sales forecasting, this competition's test trips are interleaved across the same calendar period as train. A time holdout would answer a different question.
5. Compared a constant baseline, Histogram Gradient Boosting, Random Forest, and Extra Trees using Kaggle's RMSLE metric. All three learned models use the same 150,000-trip training sample and the same untouched validation rows.
6. Selected Random Forest based on this one validation split, refit it on a 500,000-trip sample, and created `id,trip_duration` predictions for every test row.

| Model, shared 150k training sample | Train RMSLE | Validation RMSLE (lower is better) |
|---|---:|---:|
| Constant baseline | 0.79722 | 0.79661 |
| Histogram Gradient Boosting | 0.41906 | 0.42925 |
| Random Forest | 0.30396 | **0.42277** |
| Extra Trees | 0.37227 | 0.44179 |

The earlier single-model experiment trained Gradient Boosting on 300,000 trips and scored 0.42871 on validation; that result is not directly comparable to the shared-150k table. Random Forest won this comparison, but its much lower train score signals overfitting. The 500,000-trip final refit has no separate validation score. The notebook checks output row count, ID order, finite values, and nonnegative predictions.

These scores were recorded from the local run. Notebook outputs are cleared in Git to keep the repository compact; rerunning the notebook reproduces the local validation comparison.

A **late practice submission** of the selected Random Forest scored **0.41798 public / 0.41583 private** (25 October 2026), improving on the earlier Gradient Boosting submission's **0.43193 public / 0.42971 private** (23 October 2026). These are practice scores, not original 2017 contest placements.

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

Keep `model_comparison.py` beside `nyc_taxi_trip_duration.ipynb`. Locally, put `train.csv` and `test.csv` under `data/` (or their single-CSV ZIP files) and run the notebook from this folder. On Kaggle, attach the competition data and make `model_comparison.py` available in the notebook's working directory before running it. The notebook saves `output/submission_random_forest.csv` locally or `/kaggle/working/submission_random_forest.csv` on Kaggle when Random Forest wins. Raw data and predictions are ignored by Git.

For a local run, use Python 3.10+ and install `numpy`, `pandas`, `scikit-learn`, and `jupyter`. Install `pytest` too if you want to run `python -m pytest tests -q`. Kaggle notebooks already provide the modeling libraries.

The [Kaggle notebook](https://www.kaggle.com/code/punkharb/nyc-taxi-trip-duration-model-comparison) has a completed run with `submission.csv` in its output. Open that notebook and use Kaggle's **Submit to Competition** action to submit the generated file from your account. The notebook checks both common Kaggle competition-input locations before loading the files.
