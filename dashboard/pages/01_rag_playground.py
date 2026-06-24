from pathlib import Path
import sys

import pandas as pd
import streamlit as st


DASHBOARD_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(DASHBOARD_DIR))

from components.ui import inject_custom_css, hero, section_card


st.set_page_config(
    page_title="RAG Playground",
    page_icon="🧪",
    layout="wide",
)

inject_custom_css()


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATHS = [
    ROOT_DIR / "reports" / "benchmark_results.csv",
]


def find_data_file():
    for path in DATA_PATHS:
        if path.exists():
            return path
    return None


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def make_config_label(row):
    pipeline = str(row.get("pipeline", "unknown")).strip().lower()
    model = str(row.get("answer_model", "unknown")).strip().lower()
    batch_id = str(row.get("batch_id", "")).strip()
    experiment_name = str(row.get("experiment_name", "")).strip().lower()

    if pipeline == "graph_rag" and model == "openai":
        if "v2" in experiment_name or batch_id in ["exp-batch-16", "final-openai-graphrag-v2"]:
            return "GraphRAG V2 + OpenAI"
        return "GraphRAG Original + OpenAI"

    pipeline_map = {
        "vector_rag": "VectorRAG",
        "graph_rag": "GraphRAG",
        "hybrid_rag": "HybridRAG",
    }

    model_map = {
        "openai": "OpenAI",
        "nvidia": "NVIDIA",
        "groq": "Groq",
        "llama3.2:1b": "Local Llama 3.2 1B",
        "local-llama3.2-1b-ft": "Local Llama 3.2 1B FT",
        "local-llama3-2-1b-ft": "Local Llama 3.2 1B FT",
    }

    return f"{pipeline_map.get(pipeline, pipeline)} + {model_map.get(model, model)}"


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value)


hero(
    title="RAG Playground",
    subtitle=(
        "Replay benchmark questions, inspect answers, compare configurations, "
        "and debug graph/vector evidence from saved benchmark runs."
    ),
    badges=["Question Replay", "Answer Inspection", "Evidence Debugging", "GraphRAG V2"],
)


data_path = find_data_file()

if data_path is None:
    st.error("No benchmark CSV found at reports/benchmark_results.csv")
    st.stop()

df = load_data(data_path)

required_columns = [
    "pipeline",
    "answer_model",
    "question_id",
    "question",
    "answer",
    "overall_score",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()


optional_defaults = {
    "batch_id": "",
    "experiment_name": "",
    "category": "unknown",
    "generated_cypher": "",
    "graph_evidence": "",
    "vector_evidence": "",
    "evidence_used": "",
    "reason": "",
    "correctness_score": None,
    "faithfulness_score": None,
    "evidence_score": None,
    "hallucination_score": None,
    "graph_context_count": None,
    "local_latency_seconds": None,
}

for col, default in optional_defaults.items():
    if col not in df.columns:
        df[col] = default


for col in [
    "overall_score",
    "correctness_score",
    "faithfulness_score",
    "evidence_score",
    "hallucination_score",
    "graph_context_count",
    "local_latency_seconds",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


df["config"] = df.apply(make_config_label, axis=1)


st.markdown(f"**Active data file:** `{data_path}`")


# --------------------------------------------------
# Search controls
# --------------------------------------------------
st.subheader("Ask / Replay a Question")

mode = st.radio(
    "Playground mode",
    ["Benchmark Replay", "Question Search"],
    horizontal=True,
)

if mode == "Benchmark Replay":
    question_options = (
        df[["question_id", "question"]]
        .drop_duplicates()
        .sort_values("question_id")
    )

    question_label_map = {
        f"{row.question_id} — {row.question[:90]}": row.question_id
        for row in question_options.itertuples()
    }

    selected_question_label = st.selectbox(
        "Select benchmark question",
        list(question_label_map.keys()),
    )

    selected_question_id = question_label_map[selected_question_label]

    question_rows = df[df["question_id"] == selected_question_id].copy()

else:
    query = st.text_input(
        "Search question text",
        placeholder="Example: Which billionaire controls Tether?",
    )

    if not query:
        st.info("Type a search query to find matching benchmark questions.")
        st.stop()

    question_rows = df[
        df["question"].astype(str).str.contains(query, case=False, na=False)
    ].copy()

    if question_rows.empty:
        st.warning("No matching benchmark questions found.")
        st.stop()

    selected_question_id = st.selectbox(
        "Matching question IDs",
        sorted(question_rows["question_id"].unique()),
    )

    question_rows = df[df["question_id"] == selected_question_id].copy()


# --------------------------------------------------
# Configuration selection
# --------------------------------------------------
available_configs = sorted(question_rows["config"].dropna().unique())

selected_config = st.selectbox(
    "Select configuration",
    available_configs,
)

selected_rows = question_rows[question_rows["config"] == selected_config].copy()

if selected_rows.empty:
    st.warning("No row found for this configuration.")
    st.stop()

row = selected_rows.iloc[-1]


# --------------------------------------------------
# Main result
# --------------------------------------------------
st.subheader("Question")

st.markdown(
    f"""
    <div class="section-card">
        <h3>{safe_text(row["question_id"])}</h3>
        <p class="small-muted">{safe_text(row["question"])}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Overall", f"{row['overall_score']:.2f}/5" if pd.notna(row["overall_score"]) else "N/A")
col2.metric("Correctness", f"{row['correctness_score']:.2f}/5" if pd.notna(row["correctness_score"]) else "N/A")
col3.metric("Faithfulness", f"{row['faithfulness_score']:.2f}/5" if pd.notna(row["faithfulness_score"]) else "N/A")
col4.metric("Evidence", f"{row['evidence_score']:.2f}/5" if pd.notna(row["evidence_score"]) else "N/A")
col5.metric("Hallucination", f"{row['hallucination_score']:.2f}" if pd.notna(row["hallucination_score"]) else "N/A")
col6.metric("Latency", f"{row['local_latency_seconds']:.2f}s" if pd.notna(row["local_latency_seconds"]) else "N/A")


st.subheader("Answer")

st.markdown(
    f"""
    <div class="section-card">
        <p>{safe_text(row["answer"])}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Evidence tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Generated Cypher",
        "Graph Evidence",
        "Vector Evidence",
        "Evaluator Reason",
        "Compare All Configs",
    ]
)


with tab1:
    cypher = safe_text(row["generated_cypher"])

    if cypher.strip():
        st.code(cypher, language="cypher")
    else:
        st.info("No generated Cypher for this row.")


with tab2:
    graph_evidence = safe_text(row["graph_evidence"])

    st.metric(
        "Graph Context Count",
        f"{row['graph_context_count']:.0f}" if pd.notna(row["graph_context_count"]) else "N/A",
    )

    if graph_evidence.strip():
        st.text_area(
            "Graph evidence",
            graph_evidence,
            height=300,
        )
    else:
        st.info("No graph evidence for this row.")


with tab3:
    vector_evidence = safe_text(row["vector_evidence"])

    if vector_evidence.strip():
        st.text_area(
            "Vector evidence",
            vector_evidence,
            height=300,
        )
    else:
        st.info("No vector evidence for this row.")


with tab4:
    reason = safe_text(row["reason"])
    evidence_used = safe_text(row["evidence_used"])

    if reason.strip():
        st.markdown("### Evaluation Reason")
        st.write(reason)
    else:
        st.info("No evaluator reason available.")

    if evidence_used.strip():
        st.markdown("### Evidence Used")
        st.write(evidence_used)


with tab5:
    compare_cols = [
        "batch_id",
        "config",
        "pipeline",
        "answer_model",
        "overall_score",
        "correctness_score",
        "faithfulness_score",
        "evidence_score",
        "hallucination_score",
        "local_latency_seconds",
        "answer",
    ]

    compare_cols = [col for col in compare_cols if col in question_rows.columns]

    compare_df = question_rows[compare_cols].copy()
    compare_df = compare_df.sort_values("overall_score", ascending=False)

    st.dataframe(
        compare_df,
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# Final note
# --------------------------------------------------
section_card(
    "How to use this page",
    (
        "Use this page to replay benchmark questions and inspect how each configuration answered. "
        "For live RAG execution, connect this page later to your actual vector_rag, graph_rag, "
        "and hybrid_rag pipeline functions."
    ),
)