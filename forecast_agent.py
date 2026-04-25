"""
forecast_agent.py — Time Series Forecasting using XGBoost.

KEY FIX: Aggregates raw order-level data into daily/weekly totals
before forecasting. Raw order data has too much noise for good predictions.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _find_date_column(df):
    priority = ["date", "time", "week", "month", "year", "day", "period"]
    for hint in priority:
        for col in df.columns:
            if hint in col.lower():
                try:
                    pd.to_datetime(df[col].dropna().head(10))
                    return col
                except Exception:
                    pass
    for col in df.select_dtypes(include="object").columns:
        try:
            pd.to_datetime(df[col].dropna().head(20))
            return col
        except Exception:
            pass
    return None


def _find_target_column(df, date_col):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None
    priority = ["revenue", "sales", "gmv", "amount", "value", "total", "count", "price"]
    for hint in priority:
        for col in numeric_cols:
            if hint in col.lower():
                return col
    return numeric_cols[0]


def _aggregate_by_date(df, date_col, target_col):
    """
    Aggregate order-level data into a time series suitable for forecasting.

    Steps:
    1. Clip per-row outliers using IQR (prevents anomalies inflating weekly totals)
    2. Aggregate to weekly or monthly depending on date range
    3. Return a clean, smooth time series
    """
    work = df[[date_col, target_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col, target_col])

    # Clip per-row outliers before aggregating (IQR method)
    Q1  = work[target_col].quantile(0.25)
    Q3  = work[target_col].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 3 * IQR   # use 3x IQR to keep legit large orders
    work[target_col] = work[target_col].clip(upper=upper)

    work = work.groupby(date_col)[target_col].sum().reset_index()
    work = work.sort_values(date_col).reset_index(drop=True)

    if len(work) < 2:
        return work

    diffs      = work[date_col].diff().dropna()
    median_diff = diffs.median()

    # Choose aggregation frequency based on date range
    date_range_days = (work[date_col].iloc[-1] - work[date_col].iloc[0]).days

    if median_diff <= pd.Timedelta(days=2):
        if date_range_days > 180:
            # Long range daily data → monthly (smoother, more forecastable)
            work = work.set_index(date_col).resample("ME")[target_col].sum().reset_index()
        else:
            # Short range daily data → weekly
            work = work.set_index(date_col).resample("W")[target_col].sum().reset_index()

    return work


def _make_features(df, target_col, date_col, n_lags=8):
    df = df.copy()
    df["_dow"]     = df[date_col].dt.dayofweek
    df["_month"]   = df[date_col].dt.month
    df["_quarter"] = df[date_col].dt.quarter
    df["_year"]    = df[date_col].dt.year
    df["_doy"]     = df[date_col].dt.dayofyear
    df["_week"]    = df[date_col].dt.isocalendar().week.astype(int)

    for lag in range(1, n_lags + 1):
        df[f"_lag_{lag}"] = df[target_col].shift(lag)

    for window in [3, 4, 8]:
        df[f"_roll_mean_{window}"] = df[target_col].shift(1).rolling(window).mean()
        df[f"_roll_std_{window}"]  = df[target_col].shift(1).rolling(window).std()

    return df


def run_forecast(df, periods=6, date_col=None, target_col=None):
    if date_col is None:
        date_col = _find_date_column(df)
    if date_col is None:
        return pd.DataFrame(), {"error": "No date column found."}

    if target_col is None:
        target_col = _find_target_column(df, date_col)
    if target_col is None:
        return pd.DataFrame(), {"error": "No numeric target column found."}

    # ✅ Aggregate first
    work = _aggregate_by_date(df, date_col, target_col)

    if len(work) < 12:
        return pd.DataFrame(), {"error": f"Need at least 12 data points after aggregation (got {len(work)})."}

    n_lags = min(8, len(work) // 4)
    feat_df = _make_features(work, target_col, date_col, n_lags=n_lags)
    feat_df = feat_df.dropna().reset_index(drop=True)

    feature_cols = [c for c in feat_df.columns if c.startswith("_")]
    X = feat_df[feature_cols]
    y = feat_df[target_col]

    if len(X) < 8:
        return pd.DataFrame(), {"error": "Not enough rows after feature engineering."}

    split = max(4, int(len(X) * 0.8))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    test_preds = model.predict(X_test)
    mae  = float(mean_absolute_error(y_test, test_preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    mape = float(np.mean(np.abs((y_test.values - test_preds) / (np.abs(y_test.values) + 1e-9))) * 100)

    # In-sample
    all_preds = model.predict(X)

    # Infer step size for future dates
    date_diffs = work[date_col].diff().dropna()
    step = date_diffs.median()

    # Historical rows
    hist_rows = []
    for idx in range(len(feat_df)):
        orig_idx = feat_df.index[idx]
        hist_rows.append({
            "date":        work[date_col].iloc[orig_idx],
            "actual":      float(work[target_col].iloc[orig_idx]),
            "predicted":   round(float(all_preds[idx]), 2),
            "is_forecast": False
        })

    # Future forecast — recursive
    last_series = work[target_col].tolist()
    last_date   = work[date_col].iloc[-1]

    future_rows = []
    for i in range(periods):
        future_date = last_date + step * (i + 1)
        temp = pd.DataFrame({date_col: [future_date], target_col: [np.nan]})
        temp[date_col] = pd.to_datetime(temp[date_col])

        temp["_dow"]     = future_date.dayofweek
        temp["_month"]   = future_date.month
        temp["_quarter"] = future_date.quarter
        temp["_year"]    = future_date.year
        temp["_doy"]     = future_date.dayofyear
        temp["_week"]    = future_date.isocalendar()[1]

        for lag in range(1, n_lags + 1):
            temp[f"_lag_{lag}"] = last_series[-lag] if lag <= len(last_series) else np.nan
        for window in [3, 4, 8]:
            vals = last_series[-window:] if len(last_series) >= window else last_series
            temp[f"_roll_mean_{window}"] = np.mean(vals)
            temp[f"_roll_std_{window}"]  = np.std(vals) if len(vals) > 1 else 0.0

        avail = [c for c in feature_cols if c in temp.columns]
        pred_val = max(0, float(model.predict(temp[avail].fillna(0))[0]))

        future_rows.append({
            "date":        future_date,
            "actual":      None,
            "predicted":   round(pred_val, 2),
            "is_forecast": True
        })
        last_series.append(pred_val)

    forecast_df = pd.DataFrame(hist_rows + future_rows)

    summary = {
        "date_col":         date_col,
        "target_col":       target_col,
        "aggregated_rows":  len(work),
        "train_rows":       len(X_train),
        "test_rows":        len(X_test),
        "forecast_periods": periods,
        "mae":  round(mae,  2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
    }

    return forecast_df, summary


def forecast_insight(summary, forecast_df):
    if "error" in summary:
        return summary["error"]

    future = forecast_df[forecast_df["is_forecast"]]
    if future.empty:
        return "Forecast generated but no future periods found."

    avg_future  = future["predicted"].mean()
    hist        = forecast_df[~forecast_df["is_forecast"]]
    last_actual = hist["actual"].iloc[-1]
    change_pct  = round((avg_future - last_actual) / (abs(last_actual) + 1e-9) * 100, 1)
    direction   = "increase" if change_pct > 0 else "decrease"

    return (
        f"XGBoost forecast for '{summary['target_col']}' over {summary['forecast_periods']} periods "
        f"(aggregated from {summary['aggregated_rows']} data points). "
        f"Average predicted value: {avg_future:,.0f} — "
        f"a {abs(change_pct)}% {direction} from the last observed value of {last_actual:,.0f}. "
        f"Model accuracy: MAE={summary['mae']:,}, MAPE={summary['mape']}%."
    )

# (patch applied inline — see _aggregate_by_date above for the main fix)