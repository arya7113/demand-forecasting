"""Train an XGBoost demand forecast and compare it with simple baselines.

Time-based split (never shuffle time series):
  train : up to day 1885
  valid : days 1886-1913 (used to decide when to stop training)
  test  : days 1914-1941 (the final 28 days, never seen during training)

Run:  python train.py --export
"""
import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from data import load_model_table
from features import FEATURES, add_features

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "app" / "artifacts"

VALID_START, TEST_START = 1886, 1914


def wape(actual, forecast):
    """Total absolute error / total sales. 0.10 = forecast is off by 10% overall."""
    return float(np.abs(forecast - actual).sum() / actual.sum())


def evaluate(test: pd.DataFrame, forecast: np.ndarray) -> dict:
    # Dept level = total sales per department per day (what a store orders and plans staff for)
    t = test.assign(forecast=forecast)
    dept = t.groupby(["dept_id", "d"], observed=True)[["sales", "forecast"]].sum()
    return {
        "item_wape": wape(t.sales, t.forecast),
        "dept_wape": wape(dept.sales, dept.forecast),
        "mae": float(np.abs(t.forecast - t.sales).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="reg:squarederror")  # also tried count:poisson, see README
    ap.add_argument("--export", action="store_true", help="save files for the dashboard")
    args = ap.parse_args()

    df = add_features(load_model_table())
    df = df[df.d >= 1100]  # first 100 days lack 28-day history
    train = df[df.d < VALID_START]
    valid = df[(df.d >= VALID_START) & (df.d < TEST_START)]
    test = df[df.d >= TEST_START].copy()

    # Baselines: what a planner could do without ML
    before = df[df.d < TEST_START]
    avg_28 = before[before.d >= TEST_START - 28].groupby("item_id", observed=True)["sales"].mean()
    last_week = before[before.d >= TEST_START - 7].set_index(["item_id", "wday"])["sales"]
    baselines = {
        "moving_average_28d": test.item_id.map(avg_28).astype(float).to_numpy(),
        "same_day_last_week": last_week.reindex(pd.MultiIndex.from_frame(test[["item_id", "wday"]])).to_numpy(),
    }

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("demand-forecasting")
    with mlflow.start_run(run_name=args.objective):
        params = {"objective": args.objective, "max_depth": 8, "learning_rate": 0.05,
                  "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist"}
        mlflow.log_params(params)

        dtrain = xgb.DMatrix(train[FEATURES], train.sales, enable_categorical=True)
        dvalid = xgb.DMatrix(valid[FEATURES], valid.sales, enable_categorical=True)
        model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dvalid, "valid")],
                          early_stopping_rounds=50, verbose_eval=100)

        forecast = model.predict(xgb.DMatrix(test[FEATURES], enable_categorical=True),
                                 iteration_range=(0, model.best_iteration + 1)).clip(0)

        results = {"xgboost": evaluate(test, forecast)}
        for name, f in baselines.items():
            results[name] = evaluate(test, f)
        for name, m in results.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()})
        print(json.dumps(results, indent=2))

        if args.export:
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            out = test[["item_id", "dept_id", "d", "date", "sales"]].assign(forecast=forecast, **baselines)
            out.to_parquet(ARTIFACTS / "test_forecasts.parquet", index=False)
            history = df[df.d >= TEST_START - 90][["item_id", "dept_id", "d", "date", "sales"]]
            history[history.d < TEST_START].to_parquet(ARTIFACTS / "history.parquet", index=False)
            (ARTIFACTS / "metrics.json").write_text(json.dumps(results, indent=2))
            importance = pd.Series(model.get_score(importance_type="gain")).sort_values(ascending=False)
            importance.rename("gain").to_frame().to_csv(ARTIFACTS / "feature_importance.csv")


if __name__ == "__main__":
    main()
