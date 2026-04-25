import re
import pandas as pd
from sqlalchemy import create_engine, text

# Create database connection
engine = create_engine("sqlite:///data.db")


def _is_safe_query(query: str) -> bool:
    """
    ✅ FIX 5: Stricter SQL safety check.

    The old check only verified the query started with SELECT, which still
    allowed stacked queries like:
        SELECT 1; DROP TABLE uploaded_data; --

    Now we:
    1. Confirm it starts with SELECT
    2. Strip comments (-- and /* */) before checking
    3. Reject any query containing a semicolon (no stacked statements)
    4. Reject known destructive keywords as a belt-and-suspenders measure
    """
    # Strip SQL comments
    cleaned = re.sub(r"--[^\n]*", "", query)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip().lower()
    cleaned = cleaned.rstrip(";").strip()
    
    if not cleaned.startswith("select"):
        return False

    if ";" in cleaned:
        return False

    # Block dangerous keywords even inside a SELECT (e.g. subquery abuse)
    dangerous = [
        "drop ", "delete ", "insert ", "update ", "alter ",
        "create ", "truncate ", "replace ", "attach ", "detach "
    ]
    if any(kw in cleaned for kw in dangerous):
        return False

    return True


def run_query(query: str):
    if not _is_safe_query(query):
        return "❌ Only safe SELECT queries are allowed. Destructive or stacked statements are blocked."

    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        return str(e)


def get_schema():
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql(query, engine)

    schema = {}

    for table in tables["name"]:
        cols = pd.read_sql(f"PRAGMA table_info({table});", engine)
        schema[table] = cols["name"].tolist()

    return schema