from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import sys

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(DASHBOARD_DIR))

from components.ui import inject_custom_css, hero, section_card, style_plotly

st.set_page_config(
    page_title="Quality Metrics",
    page_icon="✅",
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


hero(
    title="Quality Metrics",
    subtitle="Analyze correctness, faithfulness, evidence usage, refusal behavior, and hallucination patterns across benchmark configurations.",
    badges=["Correctness", "Faithfulness", "Evidence", "Hallucination", "Refusal"],
)

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
    "overall_score",
    "correctness_score",
    "faithfulness_score",
    "evidence_score",
    "refusal_score",
    "hallucination_score",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

if "batch_id" not in df.columns:
    df["batch_id"] = ""

if "experiment_name" not in df.columns:
    df["experiment_name"] = ""

if "category" not in df.columns:
    df["category"] = "unknown"

if "reason" not in df.columns:
    df["reason"] = ""

if "question" not in df.columns:
    df["question"] = ""

if "answer" not in df.columns:
    df["answer"] = ""

score_cols = [
    "overall_score",
    "correctness_score",
    "faithfulness_score",
    "evidence_score",
    "refusal_score",
    "hallucination_score",
]

for col in score_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["config"] = df.apply(make_config_label, axis=1)


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Filters")

all_configs = sorted(df["config"].dropna().unique())
all_categories = sorted(df["category"].astype(str).dropna().unique())

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

filtered_df = df[
    df["config"].isin(selected_configs)
    & df["category"].astype(str).isin(selected_categories)
].copy()

if filtered_df.empty:
    st.warning("No rows match the selected filters.")
    st.stop()


# --------------------------------------------------
# KPI cards
# --------------------------------------------------
st.subheader("Quality Overview")

avg_overall = filtered_df["overall_score"].mean()
avg_correctness = filtered_df["correctness_score"].mean()
avg_faithfulness = filtered_df["faithfulness_score"].mean()
avg_evidence = filtered_df["evidence_score"].mean()
avg_hallucination = filtered_df["hallucination_score"].mean()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Overall", f"{avg_overall:.2f}/5")
col2.metric("Correctness", f"{avg_correctness:.2f}/5")
col3.metric("Faithfulness", f"{avg_faithfulness:.2f}/5")
col4.metric("Evidence", f"{avg_evidence:.2f}/5")
col5.metric("Hallucination", f"{avg_hallucination:.2f}")


# --------------------------------------------------
# Configuration quality summary
# --------------------------------------------------
st.subheader("Quality Metrics by Configuration")

quality_summary = (
    filtered_df
    .groupby("config", dropna=False)
    .agg(
        runs=("question_id", "count"),
        questions=("question_id", "nunique"),
        avg_overall=("overall_score", "mean"),
        avg_correctness=("correctness_score", "mean"),
        avg_faithfulness=("faithfulness_score", "mean"),
        avg_evidence=("evidence_score", "mean"),
        avg_refusal=("refusal_score", "mean"),
        avg_hallucination=("hallucination_score", "mean"),
    )
    .reset_index()
    .sort_values("avg_overall", ascending=False)
)

display_summary = quality_summary.copy()

for col in [
    "avg_overall",
    "avg_correctness",
    "avg_faithfulness",
    "avg_evidence",
    "avg_refusal",
    "avg_hallucination",
]:
    display_summary[col] = display_summary[col].round(3)

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Chart: score breakdown by configuration
# --------------------------------------------------
st.subheader("Score Breakdown")

melted = quality_summary.melt(
    id_vars=["config"],
    value_vars=[
        "avg_correctness",
        "avg_faithfulness",
        "avg_evidence",
        "avg_refusal",
    ],
    var_name="metric",
    value_name="score",
)

metric_name_map = {
    "avg_correctness": "Correctness",
    "avg_faithfulness": "Faithfulness",
    "avg_evidence": "Evidence",
    "avg_refusal": "Refusal",
}

melted["metric"] = melted["metric"].map(metric_name_map)

fig_breakdown = px.bar(
    melted,
    x="config",
    y="score",
    color="metric",
    barmode="group",
    labels={
        "config": "Configuration",
        "score": "Average score",
        "metric": "Metric",
    },
    title="Quality Metric Breakdown by Configuration",
)

fig_breakdown.update_yaxes(range=[0, 5])
fig_breakdown.update_layout(xaxis_tickangle=-35, height=650)

fig_breakdown = style_plotly(fig_breakdown)
st.plotly_chart(fig_breakdown, use_container_width=True)


# --------------------------------------------------
# Hallucination chart
# --------------------------------------------------
st.subheader("Hallucination Score by Configuration")

hallucination_df = quality_summary.sort_values("avg_hallucination", ascending=True).copy()
hallucination_df["score_label"] = hallucination_df["avg_hallucination"].round(2)

fig_hallucination = px.bar(
    hallucination_df,
    x="avg_hallucination",
    y="config",
    orientation="h",
    text="score_label",
    labels={
        "avg_hallucination": "Average hallucination score",
        "config": "Configuration",
    },
    title="Lower Hallucination Score is Better",
)

fig_hallucination.update_traces(textposition="outside")
fig_hallucination.update_layout(height=max(500, 45 * len(hallucination_df)))

fig_hallucination = style_plotly(fig_hallucination)
st.plotly_chart(fig_hallucination, use_container_width=True)


# --------------------------------------------------
# Category quality breakdown
# --------------------------------------------------
st.subheader("Quality by Question Category")

category_summary = (
    filtered_df
    .groupby(["category", "config"], dropna=False)
    .agg(
        questions=("question_id", "nunique"),
        avg_overall=("overall_score", "mean"),
        avg_correctness=("correctness_score", "mean"),
        avg_faithfulness=("faithfulness_score", "mean"),
        avg_evidence=("evidence_score", "mean"),
        avg_hallucination=("hallucination_score", "mean"),
    )
    .reset_index()
)

category_display = category_summary.copy()

for col in [
    "avg_overall",
    "avg_correctness",
    "avg_faithfulness",
    "avg_evidence",
    "avg_hallucination",
]:
    category_display[col] = category_display[col].round(3)

st.dataframe(
    category_display,
    use_container_width=True,
    hide_index=True,
)

fig_category = px.bar(
    category_summary,
    x="category",
    y="avg_overall",
    color="config",
    barmode="group",
    labels={
        "category": "Question category",
        "avg_overall": "Average overall score",
        "config": "Configuration",
    },
    title="Average Overall Score by Question Category",
)

fig_category.update_yaxes(range=[0, 5])
fig_category.update_layout(xaxis_tickangle=-35, height=650)

fig_category = style_plotly(fig_category)
st.plotly_chart(fig_category, use_container_width=True)


# --------------------------------------------------
# Failure analysis
# --------------------------------------------------
st.subheader("Low-Scoring Questions")

failure_threshold = st.slider(
    "Show rows with overall score less than or equal to:",
    min_value=1.0,
    max_value=5.0,
    value=3.0,
    step=0.5,
)

failures = filtered_df[filtered_df["overall_score"] <= failure_threshold].copy()

failure_columns = [
    "batch_id",
    "config",
    "question_id",
    "category",
    "question",
    "answer",
    "overall_score",
    "correctness_score",
    "faithfulness_score",
    "evidence_score",
    "hallucination_score",
    "reason",
]

existing_failure_columns = [col for col in failure_columns if col in failures.columns]

if failures.empty:
    st.success("No low-scoring rows found for the selected threshold.")
else:
    failures = failures.sort_values(["overall_score", "config", "question_id"])
    st.dataframe(
        failures[existing_failure_columns],
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# OpenAI GraphRAG before/after quality
# --------------------------------------------------
st.subheader("OpenAI GraphRAG Quality Improvement")

openai_graph = quality_summary[
    quality_summary["config"].isin(
        [
            "GraphRAG Original + OpenAI",
            "GraphRAG V2 + OpenAI",
        ]
    )
].copy()

if openai_graph.empty:
    st.warning("OpenAI GraphRAG original/V2 rows not found.")
else:
    openai_graph_display = openai_graph.copy()

    for col in [
        "avg_overall",
        "avg_correctness",
        "avg_faithfulness",
        "avg_evidence",
        "avg_refusal",
        "avg_hallucination",
    ]:
        openai_graph_display[col] = openai_graph_display[col].round(3)

    st.dataframe(
        openai_graph_display,
        use_container_width=True,
        hide_index=True,
    )

    graph_melted = openai_graph.melt(
        id_vars=["config"],
        value_vars=[
            "avg_overall",
            "avg_correctness",
            "avg_faithfulness",
            "avg_evidence",
        ],
        var_name="metric",
        value_name="score",
    )

    metric_map = {
        "avg_overall": "Overall",
        "avg_correctness": "Correctness",
        "avg_faithfulness": "Faithfulness",
        "avg_evidence": "Evidence",
    }

    graph_melted["metric"] = graph_melted["metric"].map(metric_map)

    fig_openai = px.bar(
        graph_melted,
        x="metric",
        y="score",
        color="config",
        barmode="group",
        labels={
            "metric": "Metric",
            "score": "Average score",
            "config": "GraphRAG version",
        },
        title="OpenAI GraphRAG Original vs V2 Quality Metrics",
    )

    fig_openai.update_yaxes(range=[0, 5])
    fig_openai = style_plotly(fig_openai)
    st.plotly_chart(fig_openai, use_container_width=True)


# --------------------------------------------------
# Download
# --------------------------------------------------
csv = display_summary.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download quality metrics summary CSV",
    data=csv,
    file_name="quality_metrics_summary.csv",
    mime="text/csv",
)