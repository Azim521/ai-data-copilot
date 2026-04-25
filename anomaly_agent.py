"""
anomaly_agent.py — Anomaly Detection using Isolation Forest.

Triggered when intent == "anomaly".
Returns the original DataFrame with an extra column:
    is_anomaly: bool
    anomaly_score: float (more negative = more anomalous)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def run_anomaly_detection(
    df: pd.DataFrame,
    contamination: float = 0.05,
    columns: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Run Isolation Forest on numeric columns of df.

    Args:
        df:            Input DataFrame
        contamination: Expected proportion of anomalies (default 5%)
        columns:       Specific columns to use. None = all numeric.

    Returns:
        (result_df, summary)
        result_df — original df + 'is_anomaly' + 'anomaly_score' columns
        summary   — stats dict for display
    """
    result_df = df.copy()

    # Select numeric columns
    if columns:
        numeric_df = df[columns].select_dtypes(include="number")
    else:
        numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty or len(numeric_df.columns) == 0:
        return result_df, {"error": "No numeric columns found for anomaly detection."}

    if len(df) < 10:
        return result_df, {"error": "Need at least 10 rows for anomaly detection."}

    # Drop rows with any NaN in selected columns for fitting
    clean = numeric_df.dropna()
    if len(clean) < 10:
        return result_df, {"error": "Too many missing values in numeric columns."}

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(clean)

    # Fit Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    model.fit(X)

    # Score all rows (fill NaN rows with 0 score = not anomalous)
    all_X = scaler.transform(numeric_df.fillna(numeric_df.median()))
    scores = model.decision_function(all_X)   # lower = more anomalous
    preds  = model.predict(all_X)             # -1 = anomaly, 1 = normal

    result_df["is_anomaly"]    = (preds == -1)
    result_df["anomaly_score"] = scores.round(4)

    # Sort: anomalies first, then by score ascending (worst first)
    result_df = result_df.sort_values(
        ["is_anomaly", "anomaly_score"],
        ascending=[False, True]
    ).reset_index(drop=True)

    anomaly_count = int((preds == -1).sum())
    total = len(df)

    # Per-column stats for anomalous rows
    anomaly_rows = result_df[result_df["is_anomaly"]]
    normal_rows  = result_df[~result_df["is_anomaly"]]

    col_stats = {}
    for col in numeric_df.columns:
        if col in result_df.columns:
            col_stats[col] = {
                "anomaly_mean": round(float(anomaly_rows[col].mean()), 3) if not anomaly_rows.empty else None,
                "normal_mean":  round(float(normal_rows[col].mean()),  3) if not normal_rows.empty  else None,
            }

    summary = {
        "total_rows":    total,
        "anomaly_count": anomaly_count,
        "anomaly_pct":   round(anomaly_count / total * 100, 2),
        "columns_used":  numeric_df.columns.tolist(),
        "col_stats":     col_stats,
    }

    return result_df, summary


def anomaly_insight(summary: dict) -> str:
    """Plain English summary of anomaly results."""
    if "error" in summary:
        return summary["error"]

    lines = [
        f"Isolation Forest scanned {summary['total_rows']:,} rows across "
        f"{len(summary['columns_used'])} numeric column(s): "
        f"{', '.join(summary['columns_used'])}.",

        f"Found {summary['anomaly_count']} anomalous rows "
        f"({summary['anomaly_pct']}% of data).",
    ]

    for col, stats in summary["col_stats"].items():
        if stats["anomaly_mean"] is not None and stats["normal_mean"] is not None:
            diff_pct = 0
            if stats["normal_mean"] != 0:
                diff_pct = round(
                    (stats["anomaly_mean"] - stats["normal_mean"])
                    / abs(stats["normal_mean"]) * 100, 1
                )
            direction = "higher" if diff_pct > 0 else "lower"
            lines.append(
                f"In '{col}', anomalous rows average {stats['anomaly_mean']:,} "
                f"vs {stats['normal_mean']:,} for normal rows "
                f"({abs(diff_pct)}% {direction})."
            )

    return " ".join(lines)