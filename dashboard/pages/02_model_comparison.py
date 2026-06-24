from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import sys

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(DASHBOARD_DIR))

from components.ui import inject_custom_css, hero, section_card, style_plotly

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
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


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def clean_text(value):
    return str(value).strip()


def make_config_label(row):
    pipeline = clean_text(row.get("pipeline", "unknown")).lower()
    answer_model = clean_text(row.get("answer_model", "unknown")).lower()
    batch_id = clean_text(row.get("batch_id", ""))
    experiment_name = clean_text(row.get("experiment_name", ""))

    exp_lower = experiment_name.lower()

    # Separate original OpenAI GraphRAG and improved OpenAI GraphRAG V2
    if pipeline == "graph_rag" and answer_model == "openai":
        if "v2" in exp_lower or batch_id in ["exp-batch-16", "final-openai-graphrag-v2"]:
            return "GraphRAG V2 + OpenAI"
        return "GraphRAG Original + OpenAI"

    # Normal readable labels
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

    pipeline_name = pipeline_map.get(pipeline, pipeline)
    model_name = model_map.get(answer_model, answer_model)

    return f"{pipeline_name} + {model_name}"


def validate_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()


def safe_mean(series):
    return pd.to_numeric(series, errors="coerce").mean()


# --------------------------------------------------
# Load data
# --------------------------------------------------
hero(
    title="Model Comparison",
    subtitle="Rank VectorRAG, GraphRAG, and HybridRAG configurations across OpenAI, NVIDIA, Groq, local, and fine-tuned models.",
    badges=["Model Ranking", "Pipeline Comparison", "GraphRAG V2", "Benchmark Scores"],
)

data_path = find_data_file()

if data_path is None:
    st.error("No benchmark CSV found inside the reports/ folder.")
    st.stop()

df = load_data(data_path)

st.success(f"Loaded data from: `{data_path}`")

required_columns = [
    "pipeline",
    "answer_model",
    "question_id",
    "overall_score",
    "correctness_score",
    "faithfulness_score",
    "evidence_score",
]

validate_columns(df, required_columns)

# Optional columns
if "batch_id" not in df.columns:
    df["batch_id"] = ""

if "experiment_name" not in df.columns:
    df["experiment_name"] = ""

if "hallucination_score" not in df.columns:
    df["hallucination_score"] = pd.NA

if "local_latency_seconds" not in df.columns:
    df["local_latency_seconds"] = pd.NA

# Normalize text columns
df["pipeline"] = df["pipeline"].astype(str).str.strip()
df["answer_model"] = df["answer_model"].astype(str).str.strip()
df["batch_id"] = df["batch_id"].astype(str).str.strip()
df["experiment_name"] = df["experiment_name"].astype(str).str.strip()

# Numeric columns
score_columns = [
    "overall_score",
    "correctness_score",
    "faithfulness_score",
    "evidence_score",
    "hallucination_score",
    "local_latency_seconds",
]

for col in score_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Create readable config label
df["config"] = df.apply(make_config_label, axis=1)


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Filters")

all_pipelines = sorted(df["pipeline"].dropna().unique())
all_models = sorted(df["answer_model"].dropna().unique())
all_configs = sorted(df["config"].dropna().unique())

selected_pipelines = st.sidebar.multiselect(
    "Pipeline",
    all_pipelines,
    default=all_pipelines,
)

selected_models = st.sidebar.multiselect(
    "Answer model",
    all_models,
    default=all_models,
)

selected_configs = st.sidebar.multiselect(
    "Configuration",
    all_configs,
    default=all_configs,
)

filtered_df = df[
    df["pipeline"].isin(selected_pipelines)
    & df["answer_model"].isin(selected_models)
    & df["config"].isin(selected_configs)
].copy()

if filtered_df.empty:
    st.warning("No rows match the selected filters.")
    st.stop()


# --------------------------------------------------
# Summary table
# --------------------------------------------------
summary = (
    filtered_df
    .groupby(["config", "pipeline", "answer_model"], dropna=False)
    .agg(
        runs=("question_id", "count"),
        questions=("question_id", "nunique"),
        avg_overall_score=("overall_score", "mean"),
        avg_correctness=("correctness_score", "mean"),
        avg_faithfulness=("faithfulness_score", "mean"),
        avg_evidence=("evidence_score", "mean"),
        avg_hallucination=("hallucination_score", "mean"),
        avg_latency_seconds=("local_latency_seconds", "mean"),
    )
    .reset_index()
    .sort_values("avg_overall_score", ascending=False)
)

# Round for display
display_summary = summary.copy()
round_cols = [
    "avg_overall_score",
    "avg_correctness",
    "avg_faithfulness",
    "avg_evidence",
    "avg_hallucination",
    "avg_latency_seconds",
]

for col in round_cols:
    display_summary[col] = display_summary[col].round(3)


# --------------------------------------------------
# KPI cards
# --------------------------------------------------
st.subheader("Benchmark Overview")

total_runs = len(filtered_df)
total_questions = filtered_df["question_id"].nunique()
total_configs = filtered_df["config"].nunique()
best_row = summary.iloc[0]
best_config = best_row["config"]
best_score = best_row["avg_overall_score"]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Runs", total_runs)
col2.metric("Unique Questions", total_questions)
col3.metric("Configurations", total_configs)
col4.metric("Best Score", f"{best_score:.2f}/5")

st.info(f"Best configuration: **{best_config}**")


# --------------------------------------------------
# Ranking table
# --------------------------------------------------
st.subheader("Configuration Ranking")

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Chart 1: Overall score by config
# --------------------------------------------------
st.subheader("Average Overall Score by Configuration")

chart_df = summary.sort_values("avg_overall_score", ascending=True).copy()
chart_df["score_label"] = chart_df["avg_overall_score"].round(2)

fig = px.bar(
    chart_df,
    x="avg_overall_score",
    y="config",
    orientation="h",
    text="score_label",
    labels={
        "avg_overall_score": "Average overall score",
        "config": "Configuration",
    },
    title="RAG Configuration Performance",
)

fig.update_xaxes(range=[0, 5])
fig.update_layout(height=max(500, 45 * len(chart_df)))
fig.update_traces(textposition="outside")

fig = style_plotly(fig)
st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Pipeline comparison
# --------------------------------------------------
st.subheader("Pipeline-Level Comparison")

pipeline_summary = (
    filtered_df
    .groupby("pipeline", dropna=False)
    .agg(
        runs=("question_id", "count"),
        questions=("question_id", "nunique"),
        avg_overall_score=("overall_score", "mean"),
        avg_correctness=("correctness_score", "mean"),
        avg_faithfulness=("faithfulness_score", "mean"),
        avg_evidence=("evidence_score", "mean"),
        avg_hallucination=("hallucination_score", "mean"),
    )
    .reset_index()
    .sort_values("avg_overall_score", ascending=False)
)

pipeline_display = pipeline_summary.copy()

for col in [
    "avg_overall_score",
    "avg_correctness",
    "avg_faithfulness",
    "avg_evidence",
    "avg_hallucination",
]:
    pipeline_display[col] = pipeline_display[col].round(3)

st.dataframe(
    pipeline_display,
    use_container_width=True,
    hide_index=True,
)

pipeline_chart_df = pipeline_summary.sort_values("avg_overall_score", ascending=True).copy()
pipeline_chart_df["score_label"] = pipeline_chart_df["avg_overall_score"].round(2)

fig_pipeline = px.bar(
    pipeline_chart_df,
    x="avg_overall_score",
    y="pipeline",
    orientation="h",
    text="score_label",
    labels={
        "avg_overall_score": "Average overall score",
        "pipeline": "Pipeline",
    },
    title="Average Score by Pipeline",
)

fig_pipeline.update_xaxes(range=[0, 5])
fig_pipeline.update_traces(textposition="outside")

fig_pipeline = style_plotly(fig_pipeline)
st.plotly_chart(fig_pipeline, use_container_width=True)


# --------------------------------------------------
# OpenAI GraphRAG before/after
# --------------------------------------------------
st.subheader("OpenAI GraphRAG Before vs After Prompt Optimization")

openai_graph = summary[
    summary["config"].isin(
        [
            "GraphRAG Original + OpenAI",
            "GraphRAG V2 + OpenAI",
        ]
    )
].copy()

if openai_graph.empty:
    st.warning("OpenAI GraphRAG original/V2 rows were not found.")
else:
    openai_graph = openai_graph.sort_values("config")

    before_after_display = openai_graph[
        [
            "config",
            "questions",
            "avg_overall_score",
            "avg_correctness",
            "avg_faithfulness",
            "avg_evidence",
            "avg_hallucination",
        ]
    ].copy()

    for col in [
        "avg_overall_score",
        "avg_correctness",
        "avg_faithfulness",
        "avg_evidence",
        "avg_hallucination",
    ]:
        before_after_display[col] = before_after_display[col].round(3)

    st.dataframe(
        before_after_display,
        use_container_width=True,
        hide_index=True,
    )

    fig_graph = px.bar(
        openai_graph,
        x="config",
        y="avg_overall_score",
        text=openai_graph["avg_overall_score"].round(2),
        labels={
            "config": "GraphRAG version",
            "avg_overall_score": "Average overall score",
        },
        title="OpenAI GraphRAG: Original vs V2",
    )

    fig_graph.update_yaxes(range=[0, 5])
    fig_graph.update_traces(textposition="outside")

    fig_graph = style_plotly(fig_graph)
    st.plotly_chart(fig_graph, use_container_width=True)

    if len(openai_graph) == 2:
        scores = dict(zip(openai_graph["config"], openai_graph["avg_overall_score"]))

        original_score = scores.get("GraphRAG Original + OpenAI")
        v2_score = scores.get("GraphRAG V2 + OpenAI")

        if original_score is not None and v2_score is not None:
            improvement = v2_score - original_score
            improvement_pct = (improvement / original_score) * 100 if original_score != 0 else 0

            col_a, col_b, col_c = st.columns(3)

            col_a.metric("Original GraphRAG", f"{original_score:.2f}/5")
            col_b.metric("GraphRAG V2", f"{v2_score:.2f}/5")
            col_c.metric(
                "Improvement",
                f"+{improvement:.2f}",
                f"{improvement_pct:.1f}%",
            )


# --------------------------------------------------
# Batch-level table
# --------------------------------------------------
st.subheader("Batch-Level Details")

batch_summary = (
    filtered_df
    .groupby(["batch_id", "experiment_name", "pipeline", "answer_model", "config"], dropna=False)
    .agg(
        rows=("question_id", "count"),
        questions=("question_id", "nunique"),
        avg_overall_score=("overall_score", "mean"),
        min_score=("overall_score", "min"),
        max_score=("overall_score", "max"),
    )
    .reset_index()
)

batch_summary["first_seen_order"] = batch_summary["batch_id"].apply(
    lambda x: int(str(x).replace("exp-batch-", "")) if str(x).startswith("exp-batch-") and str(x).replace("exp-batch-", "").isdigit() else 999
)

batch_summary = batch_summary.sort_values(["first_seen_order", "batch_id"]).drop(columns=["first_seen_order"])

for col in ["avg_overall_score", "min_score", "max_score"]:
    batch_summary[col] = batch_summary[col].round(3)

st.dataframe(
    batch_summary,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Download button
# --------------------------------------------------
csv = display_summary.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download configuration summary CSV",
    data=csv,
    file_name="model_comparison_summary.csv",
    mime="text/csv",
)