"""Turn the daily sales table into model features.

We forecast 28 days ahead. On the day we make the forecast we only know sales
up to today, so every sales-based feature looks back at least 28 days.
"""
import pandas as pd

HORIZON = 28

FEATURES = [
    # Past demand (same weekday 4, 5 and 6 weeks ago)
    "lag_28", "lag_35", "lag_42",
    # Average demand over the last week / month (as of the forecast date)
    "roll_mean_7", "roll_mean_28",
    # Price
    "sell_price", "price_change",
    # Calendar
    "wday", "month", "snap", "is_event",
    # Which product this is
    "item_id", "dept_id",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["item_id", "d"]).reset_index(drop=True)
    sales = df.groupby("item_id")["sales"]

    df["lag_28"] = sales.shift(28)
    df["lag_35"] = sales.shift(35)
    df["lag_42"] = sales.shift(42)

    known = sales.shift(HORIZON)  # latest sales value we'd actually know
    df["roll_mean_7"] = known.groupby(df["item_id"]).transform(lambda s: s.rolling(7).mean())
    df["roll_mean_28"] = known.groupby(df["item_id"]).transform(lambda s: s.rolling(28).mean())

    # Price vs last week: +0.1 means 10% more expensive, -0.2 means a 20% discount
    df["price_change"] = df["sell_price"] / df.groupby("item_id")["sell_price"].shift(7) - 1

    df["is_event"] = df["event_name_1"].notna().astype(int)  # holidays, sports, festivals

    df["item_id"] = df["item_id"].astype("category")
    df["dept_id"] = df["dept_id"].astype("category")
    return df
