"""
intent_router.py — Lightweight intent classifier.

Detects what the user wants and returns a route string.
No API call needed — pure keyword + pattern matching.

Routes:
    "sql"       → SQL Agent (show, get, list, count, total, average...)
    "forecast"  → ML Forecasting Agent (predict, forecast, next, trend...)
    "anomaly"   → Anomaly Detection Agent (outlier, anomaly, spike, unusual...)
    "eda"       → EDA Agent (profile, describe, summary, distribution...)
    "compare"   → RAG + SQL combined (vs, compare, target, beat, miss...)
"""

import re

# ── Keyword banks ─────────────────────────────────────────────────────────────

FORECAST_KEYWORDS = [
    "forecast", "predict", "prediction", "next month", "next week",
    "next year", "next quarter", "future", "upcoming", "trend",
    "projection", "project", "expected", "will be", "going to be",
    "estimate", "anticipate", "extrapolate", "time series"
]

ANOMALY_KEYWORDS = [
    "anomaly", "anomalies", "outlier", "outliers", "unusual", "abnormal",
    "spike", "spikes", "strange", "odd", "irregular", "unexpected",
    "detect", "flag", "suspicious", "deviation", "deviate"
]

EDA_KEYWORDS = [
    "profile", "profiling", "describe", "description", "summary",
    "distribution", "explore", "exploration", "eda", "overview",
    "statistics", "stats", "what does the data look like",
    "tell me about the data", "analyse the data", "analyze the data",
    "missing", "null", "correlation", "correlate", "shape", "columns",
    "data types", "dtypes", "unique values"
]

COMPARE_KEYWORDS = [
    "vs", "versus", "compare", "comparison", "against", "target",
    "beat", "miss", "missed", "exceed", "exceeded", "benchmark",
    "goal", "quota", "plan", "planned", "above target", "below target",
    "underperform", "overperform"
]

# SQL is the fallback — catches everything else
SQL_KEYWORDS = [
    "show", "get", "list", "fetch", "find", "total", "sum",
    "average", "avg", "count", "max", "min", "top", "bottom",
    "highest", "lowest", "revenue", "sales", "orders", "city",
    "region", "product", "category", "group", "by", "where"
]


# ── Scorer ────────────────────────────────────────────────────────────────────

def _score(question: str, keywords: list[str]) -> int:
    """Count how many keywords appear in the question."""
    q = question.lower()
    return sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", q))


# ── Main router ───────────────────────────────────────────────────────────────

def detect_intent(question: str) -> str:
    """
    Returns the intent string for a given question.

    Priority order matters: forecast > anomaly > eda > compare > sql
    This prevents "show forecast trend" from being routed as sql.
    """
    q = question.lower().strip()

    scores = {
        "forecast": _score(q, FORECAST_KEYWORDS),
        "anomaly":  _score(q, ANOMALY_KEYWORDS),
        "eda":      _score(q, EDA_KEYWORDS),
        "compare":  _score(q, COMPARE_KEYWORDS),
        "sql":      _score(q, SQL_KEYWORDS),
    }

    # Get the highest scoring intent
    best = max(scores, key=scores.get)

    # If nothing matched at all, default to SQL
    if scores[best] == 0:
        return "sql"

    return best


def explain_intent(intent: str) -> str:
    """Human-readable label shown in the UI status bar."""
    return {
        "sql":      "🧾 SQL Agent — querying your data",
        "forecast": "📈 Forecasting Agent — running ML prediction",
        "anomaly":  "⚠️ Anomaly Agent — scanning for outliers",
        "eda":      "🔍 EDA Agent — profiling your dataset",
        "compare":  "🎯 Compare Agent — checking against targets",
    }.get(intent, "🧾 SQL Agent — querying your data")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "Show total GMV by city",
        "Forecast next month's revenue",
        "Are there any anomalies in the sales data?",
        "Give me an overview of the dataset",
        "Compare actual sales vs target",
        "Which regions missed their quota?",
        "Predict sales for next quarter",
        "Show me outliers in the orders",
        "What does the data look like?",
        "Top 5 products by revenue",
    ]

    for q in tests:
        intent = detect_intent(q)
        print(f"  {intent:10s} ← {q}")