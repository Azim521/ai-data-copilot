import os
from openai import OpenAI
from db import get_schema
from rag import retrieve_context

# ── API Key: works locally (.env) AND on Streamlit Cloud (st.secrets) ─────────
def _get_api_key() -> str:
    # 1. Try Streamlit secrets (used on Streamlit Cloud)
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    # 2. Try environment variable (used locally with .env + python-dotenv)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    return os.environ.get("OPENAI_API_KEY", "")


client = OpenAI(api_key=_get_api_key())


def clean_sql(sql: str) -> str:
    return sql.replace("```sql", "").replace("```", "").strip()


def generate_sql(question: str) -> str:
    context = retrieve_context(question)
    schema  = get_schema()

    prompt = f"""
    You are a strict SQL data analyst.

    Database schema:
    {schema}

    Relevant business context:
    {context}

    Rules:
    - Use schema columns only
    - Use business context if useful (e.g. GMV = gmv column)
    - If a required column does not exist, return exactly: COLUMN_NOT_FOUND
    - Return only the SQL query, no explanation, no markdown fences

    Question:
    {question}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    sql = response.choices[0].message.content.strip()
    return clean_sql(sql)


def generate_insight(question: str, df) -> str:
    context = retrieve_context(question)

    prompt = f"""
    You are a senior data analyst.

    User question:
    {question}

    Business context:
    {context}

    Data sample:
    {df.head().to_string()}

    Give a short, clear business insight in 2-3 lines.
    Reference specific numbers from the data.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()