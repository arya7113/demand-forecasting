---
title: Demand Forecasting
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# 28-Day Demand Forecasting (Walmart M5)

Forecasts daily unit sales 28 days ahead for **3,049 products in Walmart store CA_1**, using the
[M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) dataset.
A planner would use this to decide how much stock to order for the next four weeks.

## Results (last 28 days, never seen during training)

WAPE = total absolute error ÷ total units sold. Lower is better.

| Method | Department-level WAPE | Item-level WAPE |
|---|---|---|
| **XGBoost** | **8.7%** | **73.7%** |
| Same day last week | 9.9% | 86.8% |
| 28-day moving average | 17.2% | 74.8% |

- At department level (where ordering and staffing happen), XGBoost has **12% lower error than the best baseline** and **about half the error of a moving average**.
- Item-level daily error is high for every method: most products sell 0–2 units a day, so single-day sales are mostly noise. Adding up forecasts cancels much of that noise, which is why department-level numbers are far better.

## How it works

1. **SQL** (`src/data.py`): raw CSVs are loaded into SQLite, then one SQL query joins sales, calendar (holidays, SNAP food-stamp days) and weekly prices.
2. **Features** (`src/features.py`): 13 features. These are sales 28/35/42 days ago, 7- and 28-day average sales, price and week-on-week price change, weekday, month, event and SNAP flags, and product/department IDs. Every sales feature looks back **at least 28 days**, so the model only uses what would be known on the day the forecast is made.
3. **Model** (`src/train.py`): one XGBoost model for all products. Time-based split: train up to day 1885, validate on the next 28 days (early stopping), test on the final 28 days.
4. **Tracking**: every run is logged to **MLflow** (parameters and metrics).
5. **Dashboard** (`app/app.py`): a Streamlit app comparing forecast vs actual sales by department or item. It runs in **Docker** on Hugging Face Spaces.

## Experiments (tracked in MLflow)

| Change | Dept WAPE | Item WAPE | Kept? |
|---|---|---|---|
| Squared-error loss (final) | 8.7% | 73.7% | ✅ |
| Poisson loss | 9.2% | 73.5% | ❌ slightly better per item, worse in total |
| Earlier version: 26 features (volatility, trends, dept momentum) + Tweedie loss | 9.6% | 72.8% | ❌ twice the features, no real gain |

The two most important features were **product ID** and **28-day average sales**. The model mainly learns each product's normal level and then adjusts for weekday, price and events.

## Run it

```bash
pip install -r requirements.txt
# download M5 data into data/ (e.g. https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip)
cd src
python data.py              # build SQLite database
python train.py --export    # train, evaluate, save dashboard files
mlflow ui --backend-store-uri sqlite:///../mlflow.db   # view experiments
cd .. && docker build -t demand-forecast . && docker run -p 7860:7860 demand-forecast
```
