"""
agents.py — Master agent orchestrator.
"""

from crewai import Agent, Task, Crew
from llm import generate_sql, generate_insight
from db import get_schema
from intent_router import detect_intent, explain_intent
from eda_agent import run_eda, eda_summary_text, load_table
from anomaly_agent import run_anomaly_detection, anomaly_insight
from forecast_agent import run_forecast, forecast_insight


def _run_task(agent, description, expected_output):
    task = Task(description=description, expected_output=expected_output, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], verbose=False)
    return str(crew.kickoff())


def _validate_schema_deterministic(question, schema):
    all_columns = set()
    for cols in schema.values():
        all_columns.update(c.lower() for c in cols)
    question_words = set(w.strip("?.,!").lower() for w in question.split())
    return len(question_words & all_columns) > 0 or len(all_columns) == 0


def _run_sql_pipeline(question, run_query, schema):
    planner  = Agent(role="Planner",     goal="Break request into steps", backstory="Analytics strategist")
    qa_agent = Agent(role="QA Reviewer", goal="Check final consistency",   backstory="Output reviewer")

    plan = _run_task(planner, f"Create concise plan for this request:\n{question}", "Short numbered plan.")

    if not _validate_schema_deterministic(question, schema):
        return plan, None, "COLUMN_NOT_FOUND", "No matching columns found for your question."

    sql = generate_sql(question)
    if "COLUMN_NOT_FOUND" in sql:
        return plan, None, "COLUMN_NOT_FOUND", "Requested column is missing from the schema."

    result = run_query(sql)
    if isinstance(result, str):
        return plan, sql, result, result

    insight = generate_insight(question, result)
    qa = _run_task(
        qa_agent,
        f"Question: {question}\nSQL: {sql}\nResult: {result.head().to_string()}\nInsight: {insight}\nConsistent?",
        "Short QA summary."
    )
    return plan, sql, result, f"{insight}\n\nQA Review:\n{qa}"


def _run_eda_pipeline(question):
    df = load_table()
    if df is None:
        return "EDA", None, "No data found. Please upload a CSV first.", "No data available."
    eda = run_eda(df)
    return "EDA Analysis", None, eda, eda_summary_text(eda)


def _run_anomaly_pipeline(question, run_query):
    # FIX: Always load full table — never partial SQL subset
    df = load_table()
    if df is None:
        return "Anomaly Detection", None, "No data available. Please upload a CSV first.", "No data."
    result_df, summary = run_anomaly_detection(df)
    return "Anomaly Detection", None, result_df, anomaly_insight(summary)


def _run_forecast_pipeline(question, run_query):
    import re
    periods = 6
    match = re.search(r"(\d+)\s*(month|week|day|quarter|year)", question.lower())
    if match:
        n, unit = int(match.group(1)), match.group(2)
        periods = n if unit in ("month", "quarter", "year") else min(n, 30)

    df = load_table()
    if df is None:
        return "Forecast", None, "No data available. Please upload a CSV first.", "No data."

    forecast_df, summary = run_forecast(df, periods=periods)
    if forecast_df.empty:
        msg = summary.get("error", "Forecast failed.")
        return "Forecast", None, msg, msg

    return "Forecast", None, forecast_df, forecast_insight(summary, forecast_df)


def run_multi_agent(question, run_query):
    intent = detect_intent(question)
    schema = get_schema()

    if intent == "eda":
        return _run_eda_pipeline(question)
    elif intent == "anomaly":
        return _run_anomaly_pipeline(question, run_query)
    elif intent == "forecast":
        return _run_forecast_pipeline(question, run_query)
    else:  # sql or compare
        return _run_sql_pipeline(question, run_query, schema)