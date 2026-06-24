from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import sys

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(DASHBOARD_DIR))

from components.ui import inject_custom_css, hero, section_card, style_plotly
st.set_page_config(
    page_title="Retrieval Analysis",
    page_icon="🔎",
    layout="wide",
)
inject_custom_css()

# --------------------------------------------------
# Paths
# --------------------------------------------------
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


def text_not_empty(value):
    if pd.isna(value):
        return False
    return str(value).strip() not in ["", "nan", "None", "null"]


# --------------------------------------------------
# Load data
# --------------------------------------------------
st.title("Retrieval Analysis")
st.caption("Inspect generated Cypher, graph evidence, vector evidence, and retrieval behavior.")

data_path = find_data_file()

if data_path is None:
    st.error("No benchmark CSV found inside reports/ folder.")
    st.stop()

df = load_data(data_path)

st.success(f"Loaded data from: `{data_path}`")

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


# --------------------------------------------------
# Optional columns
# --------------------------------------------------
optional_defaults = {
    "batch_id": "",
    "experiment_name": "",
    "category": "unknown",
    "generated_cypher": "",
    "graph_context_count": 0,
    "graph_evidence": "",
    "vector_evidence": "",
    "evidence_used": "",
    "reason": "",
    "error_type": "",
    "error_message": "",
    "correctness_score": None,
    "faithfulness_score": None,
    "evidence_score": None,
    "hallucination_score": None,
}

for col, default in optional_defaults.items():
    if col not in df.columns:
        df[col] = default


# --------------------------------------------------
# Normalize
# --------------------------------------------------
text_cols = [
    "pipeline",
    "answer_model",
    "batch_id",
    "experiment_name",
    "category",
    "question",
    "answer",
    "generated_cypher",
    "graph_evidence",
    "vector_evidence",
    "evidence_used",
    "reason",
    "error_type",
    "error_message",
]

for col in text_cols:
    df[col] = df[col].astype(str).fillna("").str.strip()

numeric_cols = [
    "overall_score",
    "correctness_score",
    "faithfulness_score",
    "evidence_score",
    "hallucination_score",
    "graph_context_count",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["config"] = df.apply(make_config_label, axis=1)

df["has_cypher"] = df["generated_cypher"].apply(text_not_empty)
df["has_graph_evidence"] = df["graph_evidence"].apply(text_not_empty)
df["has_vector_evidence"] = df["vector_evidence"].apply(text_not_empty)
df["has_error"] = df["error_type"].apply(text_not_empty) | df["error_message"].apply(text_not_empty)


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Filters")

all_configs = sorted(df["config"].dropna().unique())
all_categories = sorted(df["category"].dropna().unique())
all_pipelines = sorted(df["pipeline"].dropna().unique())

selected_configs = st.sidebar.multiselect(
    "Configuration",
    all_configs,
    default=all_configs,
)

selected_categories = st.sidebar.multiselect(
    "Question category",
    all_categories,
    default=all_categories,
)

selected_pipelines = st.sidebar.multiselect(
    "Pipeline",
    all_pipelines,
    default=all_pipelines,
)

score_range = st.sidebar.slider(
    "Overall score range",
    min_value=1.0,
    max_value=5.0,
    value=(1.0, 5.0),
    step=0.5,
)

filtered_df = df[
    df["config"].isin(selected_configs)
    & df["category"].isin(selected_categories)
    & df["pipeline"].isin(selected_pipelines)
    & df["overall_score"].between(score_range[0], score_range[1])
].copy()

if filtered_df.empty:
    st.warning("No rows match the selected filters.")
    st.stop()


# --------------------------------------------------
# KPI cards
# --------------------------------------------------
st.subheader("Retrieval Overview")

total_runs = len(filtered_df)
graph_runs = int(filtered_df["has_graph_evidence"].sum())
vector_runs = int(filtered_df["has_vector_evidence"].sum())
cypher_runs = int(filtered_df["has_cypher"].sum())
error_runs = int(filtered_df["has_error"].sum())

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Runs", total_runs)
col2.metric("Graph Evidence Rows", graph_runs)
col3.metric("Vector Evidence Rows", vector_runs)
col4.metric("Cypher Rows", cypher_runs)
col5.metric("Error Rows", error_runs)


# --------------------------------------------------
# Retrieval summary by config
# --------------------------------------------------
st.subheader("Retrieval Summary by Configuration")

retrieval_summary = (
    filtered_df
    .groupby("config", dropna=False)
    .agg(
        runs=("question_id", "count"),
        questions=("question_id", "nunique"),
        avg_score=("overall_score", "mean"),
        avg_graph_context_count=("graph_context_count", "mean"),
        cypher_rows=("has_cypher", "sum"),
        graph_evidence_rows=("has_graph_evidence", "sum"),
        vector_evidence_rows=("has_vector_evidence", "sum"),
        error_rows=("has_error", "sum"),
    )
    .reset_index()
    .sort_values("avg_score", ascending=False)
)

display_summary = retrieval_summary.copy()

for col in ["avg_score", "avg_graph_context_count"]:
    display_summary[col] = display_summary[col].round(3)

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Chart: Graph context count
# --------------------------------------------------
st.subheader("Average Graph Context Count")

graph_context_df = retrieval_summary.sort_values("avg_graph_context_count", ascending=True).copy()
graph_context_df["label"] = graph_context_df["avg_graph_context_count"].round(2)

fig_context = px.bar(
    graph_context_df,
    x="avg_graph_context_count",
    y="config",
    orientation="h",
    text="label",
    labels={
        "avg_graph_context_count": "Average graph context count",
        "config": "Configuration",
    },
    title="Average Graph Context Count by Configuration",
)

fig_context.update_traces(textposition="outside")
fig_context.update_layout(height=max(500, 45 * len(graph_context_df)))

st.plotly_chart(fig_context, use_container_width=True)


# --------------------------------------------------
# Chart: Evidence source usage
# --------------------------------------------------
st.subheader("Evidence Availability")

evidence_df = retrieval_summary[
    [
        "config",
        "graph_evidence_rows",
        "vector_evidence_rows",
        "cypher_rows",
        "error_rows",
    ]
].copy()

evidence_melted = evidence_df.melt(
    id_vars=["config"],
    value_vars=[
        "graph_evidence_rows",
        "vector_evidence_rows",
        "cypher_rows",
        "error_rows",
    ],
    var_name="evidence_type",
    value_name="rows",
)

evidence_name_map = {
    "graph_evidence_rows": "Graph evidence",
    "vector_evidence_rows": "Vector evidence",
    "cypher_rows": "Generated Cypher",
    "error_rows": "Errors",
}

evidence_melted["evidence_type"] = evidence_melted["evidence_type"].map(evidence_name_map)

fig_evidence = px.bar(
    evidence_melted,
    x="config",
    y="rows",
    color="evidence_type",
    barmode="group",
    labels={
        "config": "Configuration",
        "rows": "Rows",
        "evidence_type": "Type",
    },
    title="Evidence and Retrieval Artifact Availability",
)

fig_evidence.update_layout(xaxis_tickangle=-35, height=650)

st.plotly_chart(fig_evidence, use_container_width=True)


# --------------------------------------------------
# Cypher inspection
# --------------------------------------------------
st.subheader("Generated Cypher Inspection")

cypher_df = filtered_df[filtered_df["has_cypher"]].copy()

if cypher_df.empty:
    st.info("No generated Cypher found for the selected filters.")
else:
    cypher_display_cols = [
        "batch_id",
        "config",
        "question_id",
        "category",
        "question",
        "generated_cypher",
        "graph_context_count",
        "overall_score",
        "reason",
    ]

    cypher_display_cols = [col for col in cypher_display_cols if col in cypher_df.columns]

    cypher_df = cypher_df.sort_values(["overall_score", "config", "question_id"])

    st.dataframe(
        cypher_df[cypher_display_cols],
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# Error analysis
# --------------------------------------------------
st.subheader("Retrieval / Cypher Errors")

error_df = filtered_df[filtered_df["has_error"]].copy()

if error_df.empty:
    st.success("No retrieval or Cypher errors found for selected filters.")
else:
    error_cols = [
        "batch_id",
        "config",
        "question_id",
        "category",
        "question",
        "generated_cypher",
        "error_type",
        "error_message",
        "overall_score",
    ]

    error_cols = [col for col in error_cols if col in error_df.columns]

    st.dataframe(
        error_df[error_cols],
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# Single question deep dive
# --------------------------------------------------
st.subheader("Single Run Deep Dive")

question_options = (
    filtered_df["question_id"]
    .astype(str)
    .sort_values()
    .unique()
    .tolist()
)

selected_question_id = st.selectbox(
    "Select question_id",
    question_options,
)

question_rows = filtered_df[
    filtered_df["question_id"].astype(str) == str(selected_question_id)
].copy()

if question_rows.empty:
    st.warning("No rows found for this question.")
else:
    selected_config = st.selectbox(
        "Select configuration",
        sorted(question_rows["config"].unique()),
    )

    selected_rows = question_rows[question_rows["config"] == selected_config].copy()

    if selected_rows.empty:
        st.warning("No row found for this configuration.")
    else:
        row = selected_rows.iloc[0]

        st.markdown("### Question")
        st.write(row["question"])

        st.markdown("### Answer")
        st.write(row["answer"])

        metric_cols = st.columns(5)

        metric_cols[0].metric("Overall", f"{row['overall_score']:.2f}/5")
        metric_cols[1].metric("Correctness", f"{row['correctness_score']:.2f}/5" if pd.notna(row["correctness_score"]) else "N/A")
        metric_cols[2].metric("Faithfulness", f"{row['faithfulness_score']:.2f}/5" if pd.notna(row["faithfulness_score"]) else "N/A")
        metric_cols[3].metric("Evidence", f"{row['evidence_score']:.2f}/5" if pd.notna(row["evidence_score"]) else "N/A")
        metric_cols[4].metric("Hallucination", f"{row['hallucination_score']:.2f}" if pd.notna(row["hallucination_score"]) else "N/A")

        st.markdown("### Generated Cypher")
        if text_not_empty(row["generated_cypher"]):
            st.code(row["generated_cypher"], language="cypher")
        else:
            st.info("No Cypher for this row.")

        st.markdown("### Graph Evidence")
        if text_not_empty(row["graph_evidence"]):
            st.text_area(
                "Graph evidence",
                row["graph_evidence"],
                height=220,
            )
        else:
            st.info("No graph evidence for this row.")

        st.markdown("### Vector Evidence")
        if text_not_empty(row["vector_evidence"]):
            st.text_area(
                "Vector evidence",
                row["vector_evidence"],
                height=220,
            )
        else:
            st.info("No vector evidence for this row.")

        st.markdown("### Evaluation Reason")
        if text_not_empty(row["reason"]):
            st.write(row["reason"])
        else:
            st.info("No evaluation reason available.")

        if text_not_empty(row["error_type"]) or text_not_empty(row["error_message"]):
            st.markdown("### Error")
            st.error(f"{row['error_type']} — {row['error_message']}")


# --------------------------------------------------
# Low evidence / low context rows
# --------------------------------------------------
st.subheader("Low Evidence / Low Context Cases")

low_evidence_df = filtered_df[
    (filtered_df["evidence_score"].fillna(0) <= 3)
    | (filtered_df["graph_context_count"].fillna(0) == 0)
].copy()

if low_evidence_df.empty:
    st.success("No low-evidence or zero-context cases found for selected filters.")
else:
    low_cols = [
        "batch_id",
        "config",
        "question_id",
        "category",
        "question",
        "graph_context_count",
        "evidence_score",
        "overall_score",
        "reason",
    ]

    low_cols = [col for col in low_cols if col in low_evidence_df.columns]

    low_evidence_df = low_evidence_df.sort_values(
        ["evidence_score", "graph_context_count", "overall_score"],
        ascending=[True, True, True],
    )

    st.dataframe(
        low_evidence_df[low_cols],
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# Download
# --------------------------------------------------
csv = display_summary.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download retrieval summary CSV",
    data=csv,
    file_name="retrieval_analysis_summary.csv",
    mime="text/csv",
)