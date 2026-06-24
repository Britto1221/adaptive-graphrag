from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import sys

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(DASHBOARD_DIR))

from components.ui import inject_custom_css, hero, section_card, style_plotly

st.set_page_config(
    page_title="Latency & Cost",
    page_icon="⚡",
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


def first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


# --------------------------------------------------
# Load data
# --------------------------------------------------
hero(
    title="Latency & Cost Analysis",
    subtitle="Compare average latency, p50, p95, p99, token usage, cost fields, and quality-latency tradeoffs.",
    badges=["Latency", "P50", "P95", "P99", "Cost", "Tokens"],
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
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()


# --------------------------------------------------
# Optional columns
# --------------------------------------------------
if "batch_id" not in df.columns:
    df["batch_id"] = ""

if "experiment_name" not in df.columns:
    df["experiment_name"] = ""

if "category" not in df.columns:
    df["category"] = "unknown"

latency_col = first_existing_column(
    df,
    [
        "local_latency_seconds",
        "latency_seconds",
        "generation_latency_seconds",
        "total_latency_seconds",
    ],
)

cost_col = first_existing_column(
    df,
    [
        "cost_usd",
        "total_cost_usd",
        "estimated_cost_usd",
        "cost",
    ],
)

input_tokens_col = first_existing_column(
    df,
    [
        "input_tokens",
        "prompt_tokens",
        "total_input_tokens",
    ],
)

output_tokens_col = first_existing_column(
    df,
    [
        "output_tokens",
        "completion_tokens",
        "total_output_tokens",
    ],
)

total_tokens_col = first_existing_column(
    df,
    [
        "total_tokens",
        "tokens",
        "token_count",
    ],
)

cpu_before_col = first_existing_column(
    df,
    [
        "cpu_percent_before",
        "cpu_before",
    ],
)

cpu_after_col = first_existing_column(
    df,
    [
        "cpu_percent_after",
        "cpu_after",
        "cpu_percent",
        "avg_cpu_percent",
        "cpu_usage_percent",
    ],
)

ram_before_col = first_existing_column(
    df,
    [
        "ram_used_mb_before",
        "ram_before_mb",
    ],
)

ram_after_col = first_existing_column(
    df,
    [
        "ram_used_mb_after",
        "ram_after_mb",
        "ram_used_mb",
        "memory_used_mb",
        "ram_mb",
        "memory_mb",
    ],
)

ram_delta_col = first_existing_column(
    df,
    [
        "ram_delta_mb",
    ],
)

ram_percent_before_col = first_existing_column(
    df,
    [
        "ram_percent_before",
    ],
)

ram_percent_after_col = first_existing_column(
    df,
    [
        "ram_percent_after",
        "system_ram_percent",
    ],
)

ram_available_before_col = first_existing_column(
    df,
    [
        "ram_available_mb_before",
    ],
)

ram_available_after_col = first_existing_column(
    df,
    [
        "ram_available_mb_after",
    ],
)

# Backward-compatible names used later in the page
cpu_col = cpu_after_col
ram_col = ram_after_col


# --------------------------------------------------
# Normalize
# --------------------------------------------------
df["pipeline"] = df["pipeline"].astype(str).str.strip()
df["answer_model"] = df["answer_model"].astype(str).str.strip()
df["batch_id"] = df["batch_id"].astype(str).str.strip()
df["experiment_name"] = df["experiment_name"].astype(str).str.strip()
df["category"] = df["category"].astype(str).str.strip()

df["overall_score"] = pd.to_numeric(df["overall_score"], errors="coerce")

for col in [
    latency_col,
    cost_col,
    input_tokens_col,
    output_tokens_col,
    total_tokens_col,
    cpu_before_col,
    cpu_after_col,
    ram_before_col,
    ram_after_col,
    ram_delta_col,
    ram_percent_before_col,
    ram_percent_after_col,
    ram_available_before_col,
    ram_available_after_col,
]:
    if col is not None:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if total_tokens_col is None and input_tokens_col is not None and output_tokens_col is not None:
    df["computed_total_tokens"] = df[input_tokens_col].fillna(0) + df[output_tokens_col].fillna(0)
    total_tokens_col = "computed_total_tokens"

df["config"] = df.apply(make_config_label, axis=1)


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Filters")

all_configs = sorted(df["config"].dropna().unique())
all_categories = sorted(df["category"].dropna().unique())

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
    & df["category"].isin(selected_categories)
].copy()

if filtered_df.empty:
    st.warning("No rows match the selected filters.")
    st.stop()


# --------------------------------------------------
# Latency available check
# --------------------------------------------------
if latency_col is None:
    st.error(
        "No latency column found. Expected one of: "
        "`local_latency_seconds`, `latency_seconds`, `generation_latency_seconds`, `total_latency_seconds`."
    )
    st.stop()


# --------------------------------------------------
# KPI cards
# --------------------------------------------------
st.subheader("Runtime Overview")

avg_latency = filtered_df[latency_col].mean()
p50_latency = filtered_df[latency_col].quantile(0.50)
p95_latency = filtered_df[latency_col].quantile(0.95)
p99_latency = filtered_df[latency_col].quantile(0.99)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average Latency", f"{avg_latency:.2f}s")
col2.metric("P50 Latency", f"{p50_latency:.2f}s")
col3.metric("P95 Latency", f"{p95_latency:.2f}s")
col4.metric("P99 Latency", f"{p99_latency:.2f}s")


# --------------------------------------------------
# Latency summary by config
# --------------------------------------------------
st.subheader("Latency by Configuration")

summary_agg = {
    "runs": ("question_id", "count"),
    "questions": ("question_id", "nunique"),
    "avg_score": ("overall_score", "mean"),
    "avg_latency": (latency_col, "mean"),
    "p50_latency": (latency_col, lambda x: x.quantile(0.50)),
    "p95_latency": (latency_col, lambda x: x.quantile(0.95)),
    "p99_latency": (latency_col, lambda x: x.quantile(0.99)),
}

if cost_col is not None:
    summary_agg["avg_cost"] = (cost_col, "mean")
    summary_agg["total_cost"] = (cost_col, "sum")

if total_tokens_col is not None:
    summary_agg["avg_total_tokens"] = (total_tokens_col, "mean")
    summary_agg["total_tokens"] = (total_tokens_col, "sum")

if cpu_col is not None:
    summary_agg["avg_cpu_percent"] = (cpu_col, "mean")

if ram_col is not None:
    summary_agg["avg_ram_mb"] = (ram_col, "mean")

latency_summary = (
    filtered_df
    .groupby("config", dropna=False)
    .agg(**summary_agg)
    .reset_index()
    .sort_values("avg_latency", ascending=True)
)

display_summary = latency_summary.copy()

for col in display_summary.columns:
    if col not in ["config", "runs", "questions"]:
        display_summary[col] = pd.to_numeric(display_summary[col], errors="coerce").round(3)

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Chart: average latency
# --------------------------------------------------
st.subheader("Average Latency Chart")

latency_chart_df = latency_summary.sort_values("avg_latency", ascending=True).copy()
latency_chart_df["latency_label"] = latency_chart_df["avg_latency"].round(2)

fig_latency = px.bar(
    latency_chart_df,
    x="avg_latency",
    y="config",
    orientation="h",
    text="latency_label",
    labels={
        "avg_latency": "Average latency seconds",
        "config": "Configuration",
    },
    title="Average Latency by Configuration",
)

fig_latency.update_traces(textposition="outside")
fig_latency.update_layout(height=max(500, 45 * len(latency_chart_df)))

fig_latency = style_plotly(fig_latency)
st.plotly_chart(fig_latency, use_container_width=True)


# --------------------------------------------------
# Latency distribution
# --------------------------------------------------
st.subheader("Latency Distribution")

fig_box = px.box(
    filtered_df,
    x="config",
    y=latency_col,
    points="outliers",
    labels={
        "config": "Configuration",
        latency_col: "Latency seconds",
    },
    title="Latency Distribution by Configuration",
)

fig_box.update_layout(xaxis_tickangle=-35, height=650)

fig_box = style_plotly(fig_box)
st.plotly_chart(fig_box, use_container_width=True)


# --------------------------------------------------
# Score vs latency tradeoff
# --------------------------------------------------
st.subheader("Quality vs Latency Tradeoff")

tradeoff_df = latency_summary.copy()

fig_tradeoff = px.scatter(
    tradeoff_df,
    x="avg_latency",
    y="avg_score",
    size="runs",
    hover_name="config",
    labels={
        "avg_latency": "Average latency seconds",
        "avg_score": "Average overall score",
        "runs": "Runs",
    },
    title="Higher Quality with Lower Latency is Better",
)

fig_tradeoff.update_yaxes(range=[0, 5])

fig_tradeoff = style_plotly(fig_tradeoff)
st.plotly_chart(fig_tradeoff, use_container_width=True)


# --------------------------------------------------
# Pipeline latency
# --------------------------------------------------
st.subheader("Pipeline-Level Latency")

pipeline_latency = (
    filtered_df
    .groupby("pipeline", dropna=False)
    .agg(
        runs=("question_id", "count"),
        avg_latency=(latency_col, "mean"),
        p50_latency=(latency_col, lambda x: x.quantile(0.50)),
        p95_latency=(latency_col, lambda x: x.quantile(0.95)),
        avg_score=("overall_score", "mean"),
    )
    .reset_index()
    .sort_values("avg_latency", ascending=True)
)

pipeline_display = pipeline_latency.copy()

for col in ["avg_latency", "p50_latency", "p95_latency", "avg_score"]:
    pipeline_display[col] = pipeline_display[col].round(3)

st.dataframe(
    pipeline_display,
    use_container_width=True,
    hide_index=True,
)

fig_pipeline = px.bar(
    pipeline_latency,
    x="avg_latency",
    y="pipeline",
    orientation="h",
    text=pipeline_latency["avg_latency"].round(2),
    labels={
        "avg_latency": "Average latency seconds",
        "pipeline": "Pipeline",
    },
    title="Average Latency by Pipeline",
)

fig_pipeline.update_traces(textposition="outside")

fig_pipeline = style_plotly(fig_pipeline)
st.plotly_chart(fig_pipeline, use_container_width=True)


# --------------------------------------------------
# Cost / Token section
# --------------------------------------------------
st.subheader("Cost and Token Usage")

if cost_col is None and total_tokens_col is None:
    st.info(
        "No cost or token columns were found in the CSV. "
        "This section will activate when LangSmith/exported token and cost columns are added."
    )
else:
    cost_token_cols = ["config", "runs", "questions", "avg_score", "avg_latency"]

    if cost_col is not None:
        cost_token_cols.extend(["avg_cost", "total_cost"])

    if total_tokens_col is not None:
        cost_token_cols.extend(["avg_total_tokens", "total_tokens"])

    available_cols = [col for col in cost_token_cols if col in latency_summary.columns]

    cost_token_display = latency_summary[available_cols].copy()

    for col in cost_token_display.columns:
        if col not in ["config"]:
            cost_token_display[col] = pd.to_numeric(cost_token_display[col], errors="coerce").round(4)

    st.dataframe(
        cost_token_display,
        use_container_width=True,
        hide_index=True,
    )

    if cost_col is not None:
        cost_chart_df = latency_summary.sort_values("total_cost", ascending=True).copy()
        cost_chart_df["cost_label"] = cost_chart_df["total_cost"].round(4)

        fig_cost = px.bar(
            cost_chart_df,
            x="total_cost",
            y="config",
            orientation="h",
            text="cost_label",
            labels={
                "total_cost": "Total cost",
                "config": "Configuration",
            },
            title="Total Cost by Configuration",
        )

        fig_cost.update_traces(textposition="outside")
        fig_cost.update_layout(height=max(500, 45 * len(cost_chart_df)))

        fig_cost = style_plotly(fig_cost)
        st.plotly_chart(fig_cost, use_container_width=True)

    if total_tokens_col is not None:
        token_chart_df = latency_summary.sort_values("avg_total_tokens", ascending=True).copy()
        token_chart_df["token_label"] = token_chart_df["avg_total_tokens"].round(0)

        fig_tokens = px.bar(
            token_chart_df,
            x="avg_total_tokens",
            y="config",
            orientation="h",
            text="token_label",
            labels={
                "avg_total_tokens": "Average total tokens",
                "config": "Configuration",
            },
            title="Average Token Usage by Configuration",
        )

        fig_tokens.update_traces(textposition="outside")
        fig_tokens.update_layout(height=max(500, 45 * len(token_chart_df)))

        fig_tokens = style_plotly(fig_tokens)
        st.plotly_chart(fig_tokens, use_container_width=True)


# --------------------------------------------------
# Resource section
# --------------------------------------------------
st.subheader("Resource Usage")

if cpu_col is None and ram_col is None:
    st.info("No CPU/RAM columns found in this CSV.")
else:
    resource_cols = ["config", "runs"]

    if cpu_col is not None:
        resource_cols.append("avg_cpu_percent")

    if ram_col is not None:
        resource_cols.append("avg_ram_mb")

    resource_display = latency_summary[resource_cols].copy()

    for col in resource_display.columns:
        if col != "config":
            resource_display[col] = pd.to_numeric(resource_display[col], errors="coerce").round(3)

    st.dataframe(
        resource_display,
        use_container_width=True,
        hide_index=True,
    )

    if cpu_col is not None:
        fig_cpu = px.bar(
            latency_summary.sort_values("avg_cpu_percent", ascending=True),
            x="avg_cpu_percent",
            y="config",
            orientation="h",
            labels={
                "avg_cpu_percent": "Average CPU percent",
                "config": "Configuration",
            },
            title="Average CPU Usage by Configuration",
        )

        fig_cpu = style_plotly(fig_cpu)
        st.plotly_chart(fig_cpu, use_container_width=True)

    if ram_col is not None:
        fig_ram = px.bar(
            latency_summary.sort_values("avg_ram_mb", ascending=True),
            x="avg_ram_mb",
            y="config",
            orientation="h",
            labels={
                "avg_ram_mb": "Average RAM MB",
                "config": "Configuration",
            },
            title="Average RAM Usage by Configuration",
        )

        fig_ram = style_plotly(fig_ram)
        st.plotly_chart(fig_ram, use_container_width=True)

# --------------------------------------------------
# Batch-level RAM usage
# --------------------------------------------------
st.subheader("Batch-Level RAM Usage")

if ram_after_col is None:
    st.info(
        "No RAM usage column found. Expected `ram_used_mb_after` or similar."
    )
else:
    batch_agg = {
        "rows": ("question_id", "count"),
        "questions": ("question_id", "nunique"),
        "avg_ram_used_mb_after": (ram_after_col, "mean"),
        "max_ram_used_mb_after": (ram_after_col, "max"),
        "p95_ram_used_mb_after": (ram_after_col, lambda x: x.quantile(0.95)),
        "avg_score": ("overall_score", "mean"),
    }

    if ram_before_col is not None:
        batch_agg["avg_ram_used_mb_before"] = (ram_before_col, "mean")

    if ram_delta_col is not None:
        batch_agg["avg_ram_delta_mb"] = (ram_delta_col, "mean")
        batch_agg["max_ram_delta_mb"] = (ram_delta_col, "max")

    if ram_percent_after_col is not None:
        batch_agg["avg_ram_percent_after"] = (ram_percent_after_col, "mean")

    if cpu_after_col is not None:
        batch_agg["avg_cpu_percent_after"] = (cpu_after_col, "mean")

    batch_ram = (
        filtered_df
        .groupby(
            ["batch_id", "experiment_name", "pipeline", "answer_model", "config"],
            dropna=False,
        )
        .agg(**batch_agg)
        .reset_index()
    )

    batch_ram["batch_order"] = batch_ram["batch_id"].apply(
        lambda x: int(str(x).replace("exp-batch-", ""))
        if str(x).startswith("exp-batch-")
        and str(x).replace("exp-batch-", "").isdigit()
        else 999
    )

    batch_ram = batch_ram.sort_values(["batch_order", "batch_id"]).drop(
        columns=["batch_order"]
    )

    display_batch_ram = batch_ram.copy()

    for col in display_batch_ram.columns:
        if col not in [
            "batch_id",
            "experiment_name",
            "pipeline",
            "answer_model",
            "config",
        ]:
            display_batch_ram[col] = pd.to_numeric(
                display_batch_ram[col],
                errors="coerce",
            ).round(3)

    st.dataframe(
        display_batch_ram,
        use_container_width=True,
        hide_index=True,
    )

    fig_batch_ram = px.bar(
        batch_ram,
        x="batch_id",
        y="avg_ram_used_mb_after",
        color="config",
        text=batch_ram["avg_ram_used_mb_after"].round(1),
        labels={
            "batch_id": "Batch ID",
            "avg_ram_used_mb_after": "Average RAM used after run MB",
            "config": "Configuration",
        },
        title="Average RAM Usage by Batch",
    )

    fig_batch_ram.update_traces(textposition="outside")
    fig_batch_ram.update_layout(xaxis_tickangle=-35, height=650)

    fig_batch_ram = style_plotly(fig_batch_ram)
    st.plotly_chart(fig_batch_ram, use_container_width=True)

    if ram_delta_col is not None:
        fig_batch_delta = px.bar(
            batch_ram,
            x="batch_id",
            y="avg_ram_delta_mb",
            color="config",
            text=batch_ram["avg_ram_delta_mb"].round(1),
            labels={
                "batch_id": "Batch ID",
                "avg_ram_delta_mb": "Average RAM delta MB",
                "config": "Configuration",
            },
            title="Average RAM Delta by Batch",
        )

        fig_batch_delta.update_traces(textposition="outside")
        fig_batch_delta.update_layout(xaxis_tickangle=-35, height=650)

        fig_batch_delta = style_plotly(fig_batch_delta)
        st.plotly_chart(fig_batch_delta, use_container_width=True)

    if cpu_after_col is not None:
        fig_batch_cpu = px.bar(
            batch_ram,
            x="batch_id",
            y="avg_cpu_percent_after",
            color="config",
            text=batch_ram["avg_cpu_percent_after"].round(1),
            labels={
                "batch_id": "Batch ID",
                "avg_cpu_percent_after": "Average CPU percent after run",
                "config": "Configuration",
            },
            title="Average CPU Usage by Batch",
        )

        fig_batch_cpu.update_traces(textposition="outside")
        fig_batch_cpu.update_layout(xaxis_tickangle=-35, height=650)

        fig_batch_cpu = style_plotly(fig_batch_cpu)
        st.plotly_chart(fig_batch_cpu, use_container_width=True)
# --------------------------------------------------
# Slowest rows
# --------------------------------------------------
st.subheader("Slowest Individual Runs")

slowest_cols = [
    "batch_id",
    "config",
    "question_id",
    "category",
    "overall_score",
    latency_col,
]

if "question" in filtered_df.columns:
    slowest_cols.append("question")

slowest_cols = [col for col in slowest_cols if col in filtered_df.columns]

slowest_df = (
    filtered_df
    .sort_values(latency_col, ascending=False)
    .head(20)
    .copy()
)

st.dataframe(
    slowest_df[slowest_cols],
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Download
# --------------------------------------------------
csv = display_summary.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download latency summary CSV",
    data=csv,
    file_name="latency_cost_summary.csv",
    mime="text/csv",
)