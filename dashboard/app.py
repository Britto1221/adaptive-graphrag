from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


# Allow imports from dashboard/components
DASHBOARD_DIR = Path(__file__).resolve().parent
sys.path.append(str(DASHBOARD_DIR))

from components.ui import inject_custom_css, hero, section_card, score_badge


st.set_page_config(
    page_title="Adaptive GraphRAG Dashboard",
    page_icon="🧠",
    layout="wide",
)


inject_custom_css()


ROOT_DIR = Path(__file__).resolve().parents[1]

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
    title="Adaptive GraphRAG Benchmark Dashboard",
    subtitle=(
        "A production-style evaluation dashboard comparing VectorRAG, GraphRAG, "
        "and HybridRAG across cloud, local, and fine-tuned answer models. "
        "The dashboard highlights answer quality, hallucination behavior, retrieval evidence, "
        "latency, and GraphRAG prompt optimization impact."
    ),
    badges=[
        "VectorRAG",
        "GraphRAG",
        "HybridRAG",
        "LangSmith-ready",
        "Prompt Optimization",
        "Local LLM Benchmarking",
    ],
)


data_path = find_data_file()

if data_path is None:
    st.error("No benchmark CSV found in the reports/ folder.")
    st.stop()

df = load_data(data_path)

required_columns = [
    "pipeline",
    "answer_model",
    "question_id",
    "overall_score",
    "correctness_score",
    "faithfulness_score",
    "evidence_score",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

if "batch_id" not in df.columns:
    df["batch_id"] = ""

if "experiment_name" not in df.columns:
    df["experiment_name"] = ""

if "hallucination_score" not in df.columns:
    df["hallucination_score"] = None

if "local_latency_seconds" not in df.columns:
    df["local_latency_seconds"] = None

for col in [
    "overall_score",
    "correctness_score",
    "faithfulness_score",
    "evidence_score",
    "hallucination_score",
    "local_latency_seconds",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["config"] = df.apply(make_config_label, axis=1)


st.markdown(f"**Active data file:** `{data_path}`")


# --------------------------------------------------
# KPI cards
# --------------------------------------------------
total_runs = len(df)
total_questions = df["question_id"].nunique()
total_pipelines = df["pipeline"].nunique()
total_models = df["answer_model"].nunique()
avg_score = df["overall_score"].mean()

summary = (
    df.groupby("config", dropna=False)
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

best_config = summary.iloc[0]["config"]
best_score = summary.iloc[0]["avg_overall_score"]

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Runs", f"{total_runs:,}")
col2.metric("Questions", total_questions)
col3.metric("Pipelines", total_pipelines)
col4.metric("Models", total_models)
col5.metric("Avg Score", f"{avg_score:.2f}/5")


col_a, col_b = st.columns([1.2, 1])

with col_a:
    st.subheader("Top Benchmark Result")
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{best_config}</h3>
            <p class="small-muted">
                Best-performing configuration in the active benchmark file.
            </p>
            {score_badge(best_score)}
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.subheader("Project Claim")
    openai_graph = summary[
        summary["config"].isin(
            [
                "GraphRAG Original + OpenAI",
                "GraphRAG V2 + OpenAI",
            ]
        )
    ].copy()

    if len(openai_graph) >= 2:
        scores = dict(zip(openai_graph["config"], openai_graph["avg_overall_score"]))
        original = scores.get("GraphRAG Original + OpenAI")
        v2 = scores.get("GraphRAG V2 + OpenAI")

        if original is not None and v2 is not None:
            improvement = v2 - original
            section_card(
                "GraphRAG Prompt Optimization",
                f"OpenAI GraphRAG improved from {original:.2f}/5 to {v2:.2f}/5, "
                f"a gain of +{improvement:.2f} points after Cypher prompt optimization.",
            )
    else:
        section_card(
            "GraphRAG Prompt Optimization",
            "OpenAI GraphRAG original and V2 rows were not both found in the active file.",
        )


# --------------------------------------------------
# Main chart
# --------------------------------------------------
st.subheader("Configuration Ranking")

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
    title="Average Overall Score by Configuration",
)

fig.update_xaxes(range=[0, 5])
fig.update_layout(
    height=max(520, 45 * len(chart_df)),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.85)",
    font=dict(color="#e5e7eb"),
    title_font=dict(size=22, color="#f8fafc"),
)

fig.update_traces(textposition="outside")

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Table
# --------------------------------------------------
display_summary = summary.copy()

for col in [
    "avg_overall_score",
    "avg_correctness",
    "avg_faithfulness",
    "avg_evidence",
    "avg_hallucination",
    "avg_latency_seconds",
]:
    display_summary[col] = display_summary[col].round(3)

st.subheader("Benchmark Summary Table")

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Page guide
# --------------------------------------------------
st.subheader("Dashboard Navigation")

col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    section_card(
        "Model Comparison",
        "Rank pipeline-model combinations and compare GraphRAG original vs V2.",
    )
    section_card(
        "Retrieval Analysis",
        "Inspect Cypher, graph evidence, vector evidence, and retrieval errors.",
    )

with col_nav2:
    section_card(
        "Quality Metrics",
        "Analyze correctness, faithfulness, hallucination, evidence, and refusal scores.",
    )
    section_card(
        "Latency & Cost",
        "Compare average latency, p50, p95, p99, and cost/token fields when available.",
    )

with col_nav3:
    section_card(
        "RAG Playground",
        "Test live queries against your RAG pipelines.",
    )
    section_card(
        "System Monitor",
        "Track CPU and RAM while running experiments or demos.",
    )