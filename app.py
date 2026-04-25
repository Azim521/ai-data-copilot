import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from agents import run_multi_agent
from db import run_query, get_schema, engine
from rag import build_vector_store, build_vector_store_from_file, get_indexed_docs, load_existing_docs
from intent_router import detect_intent, explain_intent
from pdf_report import create_pdf_report

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Data Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; max-width: 1400px; }
.main-title { font-size: 2.8rem; font-weight: 800; margin-bottom: 0.1rem; }
.sub-title { color: #94a3b8; font-size: 1rem; margin-bottom: 1rem; }
.intent-badge {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.82rem;
    color: #94a3b8;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
for key, default in [
    ("last_analysis", None),
    ("messages", []),
    ("history", []),
    ("show_insight", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Load existing RAG docs on startup (deferred so API key is ready)
try:
    load_existing_docs()
except Exception:
    pass

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def clean_column_names(columns):
    cleaned = []
    for col in columns:
        col = str(col).replace("\ufeff", "").strip().lower()
        col = re.sub(r"[ /()-]+", "_", col)
        col = re.sub(r"_+", "_", col).strip("_")
        cleaned.append(col)
    final_cols, counts = [], {}
    for col in cleaned:
        if col in counts:
            counts[col] += 1
            final_cols.append(f"{col}_{counts[col]}")
        else:
            counts[col] = 0
            final_cols.append(col)
    return final_cols


def load_csv_safely(uploaded_file):
    for enc in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc, sep=None, engine="python")
        except Exception:
            continue
    return None


def smart_chart(result: pd.DataFrame, question: str = ""):
    """Pick chart type based on question intent."""
    numeric_cols = result.select_dtypes(include="number").columns.tolist()
    text_cols    = result.select_dtypes(exclude="number").columns.tolist()

    if not numeric_cols:
        st.info("No numeric columns to chart.")
        return

    q = question.lower()
    is_time = any(w in q for w in ["trend", "over time", "monthly", "weekly", "daily", "forecast"])
    is_dist = any(w in q for w in ["distribution", "histogram", "spread"])
    is_pie  = any(w in q for w in ["share", "proportion", "breakdown", "percentage", "pie"])

    if is_dist:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5 * len(numeric_cols), 3))
        if len(numeric_cols) == 1:
            axes = [axes]
        fig.patch.set_facecolor("#0f172a")
        for ax, col in zip(axes, numeric_cols):
            ax.set_facecolor("#1e293b")
            ax.hist(result[col].dropna(), bins=20, color="#6366f1", alpha=0.85, edgecolor="#334155")
            ax.set_title(col, color="#f1f5f9", fontsize=9)
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    elif is_time or (text_cols and "date" in text_cols[0].lower()):
        if text_cols:
            chart_df = result.set_index(text_cols[0])[numeric_cols]
        else:
            chart_df = result[numeric_cols]
        st.line_chart(chart_df, use_container_width=True)

    elif text_cols:
        chart_df = result.set_index(text_cols[0])[numeric_cols]
        st.bar_chart(chart_df, use_container_width=True)

    else:
        st.line_chart(result[numeric_cols], use_container_width=True)


def render_eda(eda: dict):
    """Render full EDA report in Streamlit."""
    shape = eda["shape"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",       f"{shape['rows']:,}")
    c2.metric("Columns",    shape["columns"])
    c3.metric("Duplicates", eda["duplicates"]["count"])
    c4.metric("Missing Cols", len(eda["missing"]))

    st.markdown("---")

    # Warnings
    st.subheader("🚦 Data Quality Warnings")
    for w in eda["warnings"]:
        if w.startswith("✅"):
            st.success(w)
        else:
            st.warning(w)

    # Numeric summary
    if eda["summary"]:
        st.subheader("📐 Numeric Summary")
        st.dataframe(pd.DataFrame(eda["summary"]).round(3), use_container_width=True)

    # Missing values
    if eda["missing"]:
        st.subheader("🕳️ Missing Values")
        st.dataframe(pd.DataFrame(eda["missing"]).T, use_container_width=True)

    # Correlations
    if eda["correlations"]:
        st.subheader("🔗 Correlations")
        corr_df = pd.DataFrame(eda["correlations"]).round(3)
        st.dataframe(corr_df.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                     use_container_width=True)

    # Categoricals
    if eda["categoricals"]:
        st.subheader("🏷️ Categorical Columns")
        for col, info in eda["categoricals"].items():
            with st.expander(f"{col}  ({info['unique_count']} unique values)"):
                st.bar_chart(pd.Series(info["top_values"]))

    # Date columns
    if eda["date_columns"]:
        st.info(f"📅 Date columns detected: {', '.join(eda['date_columns'])} — try asking for a forecast!")


def render_forecast(result: pd.DataFrame, insight: str):
    """Render forecast chart with historical vs predicted."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    hist = result[~result["is_forecast"]]
    fut  = result[result["is_forecast"]]

    ax.plot(hist["date"], hist["actual"],    color="#6366f1", linewidth=2,  label="Actual",    zorder=3)
    ax.plot(hist["date"], hist["predicted"], color="#22d3ee", linewidth=1.5, linestyle="--", label="Fitted", zorder=3, alpha=0.7)
    ax.plot(fut["date"],  fut["predicted"],  color="#f59e0b", linewidth=2.5, label="Forecast",  zorder=4)

    # Shade forecast region
    if not fut.empty:
        ax.axvspan(fut["date"].iloc[0], fut["date"].iloc[-1], alpha=0.08, color="#f59e0b")

    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.grid(axis="y", color="#334155", linestyle="--", linewidth=0.5)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f1f5f9", fontsize=9)
    plt.tight_layout()

    st.pyplot(fig)
    st.info(insight)

    st.subheader("📋 Forecast Data")
    st.dataframe(result, use_container_width=True)


def render_anomaly(result: pd.DataFrame, insight: str):
    """Render anomaly results with highlighted anomaly rows."""
    anomaly_count = int(result["is_anomaly"].sum()) if "is_anomaly" in result.columns else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows",   len(result))
    c2.metric("Anomalies",    anomaly_count)
    c3.metric("Normal Rows",  len(result) - anomaly_count)

    st.info(insight)

    st.subheader("🔍 Results (anomalies shown first)")

    if "is_anomaly" in result.columns:
        def highlight_anomaly(row):
            if row.get("is_anomaly", False):
                return ["background-color: #3d1515; color: #fca5a5"] * len(row)
            return [""] * len(row)

        st.dataframe(
            result.style.apply(highlight_anomaly, axis=1),
            use_container_width=True
        )
    else:
        st.dataframe(result, use_container_width=True)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.title("⚙️ Workspace")

    # CSV Upload
    st.subheader("📊 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df_upload = load_csv_safely(uploaded_file)
        if df_upload is not None:
            df_upload.columns = clean_column_names(df_upload.columns)
            df_upload.dropna(how="all", inplace=True)
            df_upload.to_sql("uploaded_data", engine, if_exists="replace", index=False)
            st.success("✅ CSV uploaded!")
            st.caption(f"Rows: {len(df_upload)} | Columns: {len(df_upload.columns)}")
        else:
            st.error("❌ Could not read CSV file.")

    st.markdown("---")

    # RAG Upload — now accepts TXT, PDF, DOCX
    st.subheader("🧠 Knowledge Base")
    st.caption("Upload TXT, PDF, or DOCX to ground AI insights")

    rag_file = st.file_uploader(
        "Upload Knowledge Document",
        type=["txt", "pdf", "docx"]
    )
    if rag_file is not None:
        # Save to rag_docs/
        os.makedirs("rag_docs", exist_ok=True)
        save_path = os.path.join("rag_docs", rag_file.name)
        with open(save_path, "wb") as f:
            f.write(rag_file.read())

        from rag import build_vector_store_from_file
        chunks = build_vector_store_from_file(save_path)
        if chunks > 0:
            st.success(f"✅ Indexed '{rag_file.name}' — {chunks} chunks")
        else:
            st.warning("⚠️ File was empty or unreadable.")

    # Show indexed docs
    indexed = get_indexed_docs()
    if indexed:
        st.caption(f"📚 Indexed: {', '.join(indexed)}")

    st.markdown("---")

    # Schema viewer
    try:
        schema = get_schema()
        if "show_schema" not in st.session_state:
            st.session_state.show_schema = False
        if st.button("📌 Show Schema"):
            st.session_state.show_schema = not st.session_state.show_schema
        if st.session_state.show_schema:
            st.json(schema)
    except Exception:
        pass

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.last_analysis = None
        st.session_state.show_insight = False
        st.success("Cleared.")

# ---------------------------------------------------
# HERO
# ---------------------------------------------------
st.markdown('<div class="main-title">🤖 AI Data Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">SQL • EDA • Forecasting • Anomaly Detection • RAG • Multi-Agent Intelligence</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------
question = st.chat_input("Ask anything... 'Show GMV by city' • 'Forecast next 6 months' • 'Any anomalies?' • 'Profile the data'")

# ---------------------------------------------------
# RUN ANALYSIS
# ---------------------------------------------------
if question:
    st.session_state.show_insight = False

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        intent  = detect_intent(question)
        label   = explain_intent(intent)

        st.markdown(f'<div class="intent-badge">{label}</div>', unsafe_allow_html=True)

        progress = st.progress(0)
        status   = st.empty()
        status.info(f"{label}...")
        progress.progress(15)

        plan, sql, result, insight = run_multi_agent(question, run_query)

        progress.progress(100)
        status.success("✅ Done!")
        progress.empty()
        status.empty()

        st.session_state.last_analysis = {
            "question": question,
            "intent":   intent,
            "plan":     plan,
            "sql":      sql,
            "result":   result,
            "insight":  insight,
        }

        # Quick error check
        if isinstance(result, str):
            st.error(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Analysis complete ✅"})

    st.session_state.history.append(question)

# ---------------------------------------------------
# DISPLAY RESULTS — always rendered from session_state
# ---------------------------------------------------
if st.session_state.last_analysis:
    data    = st.session_state.last_analysis
    result  = data["result"]
    sql     = data["sql"]
    plan    = data["plan"]
    insight = data["insight"]
    q_text  = data["question"]
    intent  = data.get("intent", "sql")

    if isinstance(result, str):
        pass  # already shown as error above

    elif isinstance(result, dict):
        # ── EDA result ────────────────────────────────────────────────────────
        st.subheader("🔍 Exploratory Data Analysis")
        render_eda(result)
        st.markdown("---")
        st.caption(insight)

    elif isinstance(result, pd.DataFrame):
        is_forecast = intent == "forecast" and "is_forecast" in result.columns
        is_anomaly  = intent == "anomaly"  and "is_anomaly"  in result.columns

        if is_forecast:
            # ── Forecast view ─────────────────────────────────────────────────
            st.subheader("📈 Forecast")
            render_forecast(result, insight)

        elif is_anomaly:
            # ── Anomaly view ──────────────────────────────────────────────────
            st.subheader("⚠️ Anomaly Detection")
            render_anomaly(result, insight)

        else:
            # ── Standard SQL result ───────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows",    len(result))
            c2.metric("Columns", len(result.columns))
            c3.metric("Status",  "Success")
            c4.metric("Intent",  intent.upper())

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Results", "📈 Chart", "🧾 SQL",
                "🧠 Reasoning", "💡 Insight", "📄 Export"
            ])

            with tab1:
                st.dataframe(result, use_container_width=True)

            with tab2:
                smart_chart(result, q_text)

            with tab3:
                st.code(sql or "No SQL generated.", language="sql")

            with tab4:
                st.write(plan)

            with tab5:
                if st.button("💡 Generate AI Insight"):
                    st.session_state.show_insight = True
                if st.session_state.show_insight:
                    st.success(insight)
                else:
                    st.info("Click the button to generate insight.")

            with tab6:
                if sql:
                    pdf_path = create_pdf_report(q_text, sql, result, insight)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF Report",
                            data=f,
                            file_name="AI_Report.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.info("PDF export available for SQL queries only.")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit • CrewAI • XGBoost • Isolation Forest • FAISS RAG • Multi-Agent Intelligence")