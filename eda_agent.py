"""
eda_agent.py — Automatic Exploratory Data Analysis.

Triggered when intent == "eda" or on CSV upload.
No API call needed — pure pandas analysis.

Returns a dict with all EDA sections so app.py can render
each section in its own UI component.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from db import engine


def load_table(table_name: str = "uploaded_data") -> pd.DataFrame | None:
    """Load a table from the SQLite database."""
    try:
        return pd.read_sql(f"SELECT * FROM {table_name}", engine)
    except Exception:
        return None


def run_eda(df: pd.DataFrame) -> dict:
    """
    Run full EDA on a DataFrame.

    Returns a dict with keys:
        shape, dtypes, missing, summary, correlations,
        categoricals, top_values, duplicates, warnings
    """
    result = {}

    # ── Shape ─────────────────────────────────────────────────────────────────
    result["shape"] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_cells": int(df.shape[0] * df.shape[1])
    }

    # ── Data types ────────────────────────────────────────────────────────────
    result["dtypes"] = df.dtypes.astype(str).to_dict()

    # ── Missing values ────────────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    result["missing"] = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    }).query("missing_count > 0").to_dict(orient="index")

    # ── Numeric summary ───────────────────────────────────────────────────────
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        desc = numeric_df.describe().round(3)
        # Add skewness and kurtosis
        desc.loc["skewness"] = numeric_df.skew().round(3)
        desc.loc["kurtosis"] = numeric_df.kurtosis().round(3)
        result["summary"] = desc.to_dict()
    else:
        result["summary"] = {}

    # ── Correlations ──────────────────────────────────────────────────────────
    if len(numeric_df.columns) >= 2:
        result["correlations"] = numeric_df.corr().round(3).to_dict()
    else:
        result["correlations"] = {}

    # ── Categorical columns ───────────────────────────────────────────────────
    cat_df = df.select_dtypes(exclude="number")
    cat_info = {}
    for col in cat_df.columns:
        vc = df[col].value_counts()
        cat_info[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": vc.head(5).to_dict(),
            "sample": df[col].dropna().head(3).tolist()
        }
    result["categoricals"] = cat_info

    # ── Duplicate rows ────────────────────────────────────────────────────────
    dup_count = int(df.duplicated().sum())
    result["duplicates"] = {
        "count": dup_count,
        "pct": round(dup_count / len(df) * 100, 2)
    }

    # ── Auto warnings ─────────────────────────────────────────────────────────
    warnings = []

    # High missing
    for col, info in result["missing"].items():
        if info["missing_pct"] > 30:
            warnings.append(f"⚠️ '{col}' has {info['missing_pct']}% missing values")

    # High cardinality categoricals
    for col, info in cat_info.items():
        if info["unique_count"] > 50:
            warnings.append(f"⚠️ '{col}' has {info['unique_count']} unique values — consider grouping")

    # Duplicate rows
    if dup_count > 0:
        warnings.append(f"⚠️ {dup_count} duplicate rows found ({result['duplicates']['pct']}%)")

    # Highly skewed numerics
    if result["summary"]:
        for col, stats in result["summary"].items():
            skew = stats.get("skewness", 0)
            if abs(skew) > 2:
                warnings.append(f"⚠️ '{col}' is highly skewed (skewness={skew}) — consider log transform")

    # High correlations
    if result["correlations"]:
        corr_df = pd.DataFrame(result["correlations"])
        for i in range(len(corr_df.columns)):
            for j in range(i + 1, len(corr_df.columns)):
                c1 = corr_df.columns[i]
                c2 = corr_df.columns[j]
                val = corr_df.iloc[i, j]
                if abs(val) > 0.9:
                    warnings.append(
                        f"⚠️ '{c1}' and '{c2}' are highly correlated ({val}) — possible redundancy"
                    )

    result["warnings"] = warnings if warnings else ["✅ No major data quality issues detected"]

    # ── Date columns ──────────────────────────────────────────────────────────
    date_cols = []
    for col in df.columns:
        if df[col].dtype == object:
            try:
                pd.to_datetime(df[col].dropna().head(20))
                date_cols.append(col)
            except Exception:
                pass
    result["date_columns"] = date_cols

    return result


def eda_summary_text(eda: dict) -> str:
    """
    One-paragraph plain English summary of the EDA result.
    Used for the insight section and PDF report.
    """
    shape = eda["shape"]
    missing_count = len(eda["missing"])
    dup = eda["duplicates"]["count"]
    num_cols = len(eda["summary"])
    cat_cols = len(eda["categoricals"])
    warnings = [w for w in eda["warnings"] if w.startswith("⚠️")]

    lines = [
        f"The dataset has {shape['rows']:,} rows and {shape['columns']} columns "
        f"({num_cols} numeric, {cat_cols} categorical)."
    ]

    if missing_count:
        lines.append(f"{missing_count} column(s) have missing values.")
    if dup:
        lines.append(f"{dup} duplicate rows were found.")
    if warnings:
        lines.append(f"{len(warnings)} data quality warning(s) were raised.")
    else:
        lines.append("No major data quality issues were detected.")

    return " ".join(lines)