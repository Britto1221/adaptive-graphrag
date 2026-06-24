import streamlit as st


def inject_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #020617 100%);
            color: #e5e7eb;
        }

        footer {
            visibility: hidden;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.2);
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb;
        }

        h1, h2, h3 {
            color: #f8fafc;
            letter-spacing: -0.03em;
        }

        h1 {
            font-size: 2.4rem !important;
            font-weight: 800 !important;
        }

        h2 {
            font-size: 1.5rem !important;
            font-weight: 750 !important;
            margin-top: 1.5rem !important;
        }

        [data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.25);
        }

        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.25);
            padding: 18px;
            border-radius: 18px;
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.22);
        }

        [data-testid="stMetricLabel"] {
            color: #94a3b8;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: #f8fafc;
            font-weight: 800;
        }

        [data-testid="stAlert"] {
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }

        .js-plotly-plot {
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
        }

        .stDownloadButton button,
        .stButton button {
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 0.65rem 1.1rem;
            font-weight: 700;
            transition: 0.2s ease-in-out;
        }

        .stDownloadButton button:hover,
        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 30px rgba(37, 99, 235, 0.35);
        }

        .hero-card {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.35), transparent 30%),
                radial-gradient(circle at bottom right, rgba(124, 58, 237, 0.35), transparent 30%),
                rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.28);
            padding: 32px;
            border-radius: 28px;
            box-shadow: 0 25px 70px rgba(0, 0, 0, 0.35);
            margin-bottom: 28px;
        }

        .hero-title {
            font-size: 2.6rem;
            line-height: 1.05;
            font-weight: 900;
            color: #f8fafc;
            margin-bottom: 12px;
            letter-spacing: -0.05em;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #cbd5e1;
            max-width: 900px;
            line-height: 1.7;
        }

        .badge-row {
            margin-top: 22px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .badge {
            background: rgba(30, 41, 59, 0.9);
            color: #bfdbfe;
            border: 1px solid rgba(96, 165, 250, 0.3);
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
        }

        .section-card {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.22);
            padding: 22px;
            border-radius: 22px;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.20);
            margin-bottom: 18px;
        }

        .small-muted {
            color: #94a3b8;
            font-size: 0.92rem;
        }

        .score-pill-good {
            background: rgba(34, 197, 94, 0.15);
            color: #86efac;
            border: 1px solid rgba(34, 197, 94, 0.35);
            padding: 6px 10px;
            border-radius: 999px;
            font-weight: 800;
            font-size: 0.85rem;
        }

        .score-pill-mid {
            background: rgba(234, 179, 8, 0.15);
            color: #fde68a;
            border: 1px solid rgba(234, 179, 8, 0.35);
            padding: 6px 10px;
            border-radius: 999px;
            font-weight: 800;
            font-size: 0.85rem;
        }

        .score-pill-bad {
            background: rgba(239, 68, 68, 0.15);
            color: #fca5a5;
            border: 1px solid rgba(239, 68, 68, 0.35);
            padding: 6px 10px;
            border-radius: 999px;
            font-weight: 800;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, badges=None):
    if badges is None:
        badges = []

    badge_html = "".join([f"<span class='badge'>{badge}</span>" for badge in badges])

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            <div class="badge-row">
                {badge_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_card(title: str, body: str):
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{title}</h3>
            <p class="small-muted">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_badge(score: float):
    if score >= 4.5:
        css_class = "score-pill-good"
    elif score >= 3.0:
        css_class = "score-pill-mid"
    else:
        css_class = "score-pill-bad"

    return f"<span class='{css_class}'>{score:.2f}/5</span>"


def style_plotly(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.85)",
        font=dict(color="#e5e7eb"),
        title_font=dict(size=22, color="#f8fafc"),
        legend=dict(
            bgcolor="rgba(15,23,42,0.3)",
            bordercolor="rgba(148,163,184,0.25)",
            borderwidth=1,
        ),
        margin=dict(l=40, r=40, t=70, b=40),
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.18)",
        zerolinecolor="rgba(148,163,184,0.25)",
        color="#cbd5e1",
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.18)",
        zerolinecolor="rgba(148,163,184,0.25)",
        color="#cbd5e1",
    )

    if height is not None:
        fig.update_layout(height=height)

    return fig