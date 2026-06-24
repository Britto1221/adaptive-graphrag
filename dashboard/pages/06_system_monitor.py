from pathlib import Path
import sys
import time

import pandas as pd
import psutil
import streamlit as st
import plotly.express as px


DASHBOARD_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(DASHBOARD_DIR))

from components.ui import inject_custom_css, hero, section_card, style_plotly


st.set_page_config(
    page_title="System Monitor",
    page_icon="🖥️",
    layout="wide",
)

inject_custom_css()


hero(
    title="System Monitor",
    subtitle=(
        "Monitor CPU, RAM, disk usage, and active Python processes while running "
        "RAG benchmarks, local LLM inference, or dashboard demos."
    ),
    badges=["CPU", "RAM", "Disk", "Python Processes", "Local LLM Monitoring"],
)


# --------------------------------------------------
# Controls
# --------------------------------------------------
st.sidebar.header("Monitor Controls")

refresh_seconds = st.sidebar.slider(
    "Refresh interval seconds",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
)

auto_refresh = st.sidebar.checkbox("Auto refresh", value=False)


# --------------------------------------------------
# System metrics
# --------------------------------------------------
cpu_percent = psutil.cpu_percent(interval=0.5)
ram = psutil.virtual_memory()
disk = psutil.disk_usage("/")

ram_used_gb = ram.used / (1024 ** 3)
ram_total_gb = ram.total / (1024 ** 3)

disk_used_gb = disk.used / (1024 ** 3)
disk_total_gb = disk.total / (1024 ** 3)


st.subheader("Live System Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("CPU Usage", f"{cpu_percent:.1f}%")
col2.metric("RAM Usage", f"{ram.percent:.1f}%")
col3.metric("RAM Used", f"{ram_used_gb:.2f} / {ram_total_gb:.2f} GB")
col4.metric("Disk Usage", f"{disk.percent:.1f}%")


# --------------------------------------------------
# Resource chart
# --------------------------------------------------
resource_df = pd.DataFrame(
    {
        "Resource": ["CPU", "RAM", "Disk"],
        "Usage Percent": [cpu_percent, ram.percent, disk.percent],
    }
)

fig_resource = px.bar(
    resource_df,
    x="Resource",
    y="Usage Percent",
    text=resource_df["Usage Percent"].round(1),
    title="Current Resource Usage",
)

fig_resource.update_yaxes(range=[0, 100])
fig_resource.update_traces(textposition="outside")

fig_resource = style_plotly(fig_resource)
st.plotly_chart(fig_resource, use_container_width=True)


# --------------------------------------------------
# Memory details
# --------------------------------------------------
st.subheader("Memory Details")

memory_df = pd.DataFrame(
    {
        "Metric": [
            "Total RAM GB",
            "Used RAM GB",
            "Available RAM GB",
            "RAM Percent",
            "Total Disk GB",
            "Used Disk GB",
            "Free Disk GB",
            "Disk Percent",
        ],
        "Value": [
            round(ram_total_gb, 2),
            round(ram_used_gb, 2),
            round(ram.available / (1024 ** 3), 2),
            round(ram.percent, 2),
            round(disk_total_gb, 2),
            round(disk_used_gb, 2),
            round(disk.free / (1024 ** 3), 2),
            round(disk.percent, 2),
        ],
    }
)

st.dataframe(
    memory_df,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Python process monitor
# --------------------------------------------------
st.subheader("Active Python / Streamlit Processes")

process_rows = []

for proc in psutil.process_iter(
    attrs=["pid", "name", "cpu_percent", "memory_percent", "cmdline"]
):
    try:
        info = proc.info
        name = str(info.get("name", "")).lower()
        cmdline = " ".join(info.get("cmdline") or [])

        if (
            "python" in name
            or "streamlit" in cmdline.lower()
            or "ollama" in name
            or "ollama" in cmdline.lower()
        ):
            process_rows.append(
                {
                    "pid": info.get("pid"),
                    "name": info.get("name"),
                    "cpu_percent": info.get("cpu_percent"),
                    "memory_percent": round(info.get("memory_percent", 0), 3),
                    "cmdline": cmdline[:250],
                }
            )

    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        continue


process_df = pd.DataFrame(process_rows)

if process_df.empty:
    st.info("No Python, Streamlit, or Ollama processes found.")
else:
    process_df = process_df.sort_values(
        ["memory_percent", "cpu_percent"],
        ascending=False,
    )

    st.dataframe(
        process_df,
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# Benchmark health hints
# --------------------------------------------------
st.subheader("Benchmark Health Guide")

col_a, col_b, col_c = st.columns(3)

with col_a:
    section_card(
        "CPU Warning",
        "If CPU stays above 90% for long periods, local LLM runs may slow down heavily.",
    )

with col_b:
    section_card(
        "RAM Warning",
        "If RAM usage crosses 85%, reduce concurrent runs or use smaller local models.",
    )

with col_c:
    section_card(
        "Disk Warning",
        "If disk usage is high, clean old model files, benchmark logs, or duplicate CSV exports.",
    )


# --------------------------------------------------
# Auto refresh
# --------------------------------------------------
if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()