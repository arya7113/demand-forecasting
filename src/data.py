"""Load raw M5 CSVs into SQLite and build the modelling table with SQL."""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
DB = ROOT / "data" / "m5.db"

STORE = "CA_1"
FIRST_DAY = 1000  # keep ~2.6 years of history to fit in memory and train quickly


def build_db():
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    sales = sales[sales.store_id == STORE]
    day_cols = [c for c in sales.columns if c.startswith("d_") and int(c[2:]) >= FIRST_DAY]
    long = sales.melt(
        id_vars=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=day_cols, var_name="d", value_name="sales",
    )
    long["d"] = long["d"].str[2:].astype(int)

    cal = pd.read_csv(RAW / "calendar.csv")
    cal.insert(0, "d", range(1, len(cal) + 1))

    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[prices.store_id == STORE]

    DB.unlink(missing_ok=True)
    with sqlite3.connect(DB) as con:
        long.to_sql("sales", con, index=False)
        cal.to_sql("calendar", con, index=False)
        prices.to_sql("prices", con, index=False)
        con.executescript("""
            CREATE INDEX idx_sales ON sales(item_id, d);
            CREATE INDEX idx_cal ON calendar(d);
            CREATE INDEX idx_prices ON prices(item_id, wm_yr_wk);
        """)
    print(f"sales rows: {len(long):,}  items: {long.item_id.nunique():,}")


MODEL_TABLE_SQL = """
SELECT s.item_id, s.dept_id, s.cat_id, s.d, s.sales,
       c.date, c.wday, c.month, c.year,
       c.event_name_1, c.event_type_1, c.snap_CA AS snap,
       p.sell_price
FROM sales s
JOIN calendar c ON c.d = s.d
LEFT JOIN prices p ON p.item_id = s.item_id AND p.wm_yr_wk = c.wm_yr_wk
ORDER BY s.item_id, s.d
"""

# Items that were not yet on the shelf have no price for that week; drop those rows.
LAUNCHED_FILTER = "WHERE sell_price IS NOT NULL"


def load_model_table() -> pd.DataFrame:
    with sqlite3.connect(DB) as con:
        df = pd.read_sql(f"SELECT * FROM ({MODEL_TABLE_SQL}) {LAUNCHED_FILTER}", con)
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    build_db()
    df = load_model_table()
    print(df.shape)
    print(df.head())
