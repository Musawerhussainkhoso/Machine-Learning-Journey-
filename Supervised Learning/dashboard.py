import os
import io
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Insurance Cost Intelligence | Linear Regression Studio",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# FINANCIAL CONSTANTS & CURRENCY FORMATTERS
# ============================================================
USD_TO_PKR = 278.50  # State Bank of Pakistan benchmark exchange rate reference


def to_pkr(usd):
    return float(usd) * USD_TO_PKR


def fmt_usd(usd):
    return f"${usd:,.0f}"


def fmt_pkr(usd):
    return f"PKR {to_pkr(usd):,.0f}"


def fmt_dual(usd):
    return f"{fmt_usd(usd)}  |  {fmt_pkr(usd)}"


def fmt_money(usd, mode="dual"):
    if mode == "usd":
        return fmt_usd(usd)
    elif mode == "pkr":
        return fmt_pkr(usd)
    return fmt_dual(usd)


# ============================================================
# THEME SYSTEM
# ============================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "currency_mode" not in st.session_state:
    st.session_state.currency_mode = "dual"  # 'dual', 'usd', 'pkr'

# Vibrant Design Palette
RED = "#FF2E63"      # Neon Coral (Highlight / Smokers)
SKY = "#00C2FF"      # Electric Cyan (Non-smokers / Primary)
AMBER = "#FFB800"    # Radiant Amber (Reference lines / Caution)
PURPLE = "#8B5CF6"   # Iris Violet (Secondary accent / Northeast)
MINT = "#10E5B0"     # Emerald Mint (Positive / Southwest)
PINK = "#FF6EC7"     # Rose Pink (Female / Accent)

SMOKER_MAP = {"yes": RED, "no": SKY}
REGION_MAP = {
    "northeast": PURPLE,
    "northwest": SKY,
    "southeast": RED,
    "southwest": MINT,
}


def get_theme_tokens():
    if st.session_state.theme == "dark":
        return dict(
            bg="radial-gradient(circle at 15% 15%, #180928 0%, #080b18 40%, #0d122e 70%, #070914 100%)",
            card="rgba(255, 255, 255, 0.05)",
            card_hover="rgba(255, 255, 255, 0.09)",
            card_border="rgba(255, 255, 255, 0.12)",
            card_border_glow="rgba(255, 46, 99, 0.45)",
            text="#F8FAFC",
            sub="#94A3B8",
            hero_from="#FF2E63",
            hero_mid="#7F1032",
            hero_to="#1E1B4B",
            sidebar="linear-gradient(180deg, #120819 0%, #090c1f 100%)",
            sidebar_border="rgba(255, 46, 99, 0.35)",
            plotly_tpl="plotly_dark",
            ticker_bg="rgba(255, 255, 255, 0.04)",
            ticker_border="rgba(255, 255, 255, 0.1)",
            ticker_text="#FDE2E8",
            scale=["#0f172a", "#1e1b4b", "#4c1d95", "#8B5CF6", "#FF2E63", "#FFB800"],
            accent="rgba(255, 46, 99, 0.12)",
            grid="rgba(148, 163, 184, 0.12)",
            plot_bg="rgba(255, 255, 255, 0.015)",
        )
    return dict(
        bg="radial-gradient(circle at 15% 15%, #FFF0F5 0%, #F4F7FF 40%, #EFF6FF 70%, #FFF5F7 100%)",
        card="#FFFFFF",
        card_hover="#FFFFFF",
        card_border="rgba(255, 46, 99, 0.18)",
        card_border_glow="rgba(255, 46, 99, 0.4)",
        text="#0F172A",
        sub="#475569",
        hero_from="#FF2E63",
        hero_mid="#B3123C",
        hero_to="#6366F1",
        sidebar="#FFFFFF",
        sidebar_border="rgba(255, 46, 99, 0.2)",
        plotly_tpl="plotly_white",
        ticker_bg="#FFFFFF",
        ticker_border="rgba(255, 46, 99, 0.15)",
        ticker_text="#7F1032",
        scale=["#dbeafe", "#93c5fd", "#6366f1", "#8B5CF6", "#FF2E63", "#FFB800"],
        accent="rgba(255, 46, 99, 0.08)",
        grid="rgba(100, 116, 139, 0.18)",
        plot_bg="#F8FAFC",
    )


th = get_theme_tokens()

# ============================================================
# GLOBAL CSS INJECTION
# ============================================================
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}
.stApp {{
    background: {th['bg']};
    background-attachment: fixed;
}}
.stApp::before, .stApp::after {{
    content: '';
    position: fixed;
    border-radius: 50%;
    filter: blur(140px);
    z-index: 0;
    pointer-events: none;
    opacity: 0.22;
}}
.stApp::before {{
    width: 500px;
    height: 500px;
    background: #FF2E63;
    top: -150px;
    left: -130px;
}}
.stApp::after {{
    width: 540px;
    height: 540px;
    background: #00C2FF;
    bottom: -180px;
    right: -140px;
}}

#MainMenu, footer {{ visibility: hidden; height: 0; }}

button[data-testid="collapsedControl"] {{
    background: linear-gradient(135deg, #FF2E63, #7F1032) !important;
    color: #fff !important;
    border-radius: 0 12px 12px 0 !important;
    box-shadow: 0 8px 24px rgba(255, 46, 99, 0.5) !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
    height: 60px !important;
    width: 42px !important;
    top: 45% !important;
}}
button[data-testid="collapsedControl"]:hover {{
    width: 52px !important;
    box-shadow: 0 0 32px rgba(255, 46, 99, 0.85) !important;
}}
button[data-testid="collapsedControl"] svg {{ fill: #fff !important; }}

header[data-testid="stHeader"] {{
    background: transparent !important;
    visibility: visible !important;
    height: 2.2rem !important;
}}

.block-container {{
    padding-top: 1rem;
    padding-bottom: 2.5rem;
    max-width: 1540px;
}}

section[data-testid="stSidebar"] {{
    background: {th['sidebar']} !important;
    border-right: 1px solid {th['sidebar_border']};
    box-shadow: 6px 0 35px rgba(255, 46, 99, 0.08);
}}
section[data-testid="stSidebar"] * {{
    color: {th['text']} !important;
}}

.side-title {{
    font-weight: 900;
    letter-spacing: 2.2px;
    font-size: 0.85rem;
    padding: 13px 16px;
    border-radius: 14px;
    background: linear-gradient(120deg, #FF2E63, #7F1032);
    color: #fff !important;
    margin-bottom: 14px;
    text-align: center;
    box-shadow: 0 8px 26px rgba(255, 46, 99, 0.45);
    position: relative;
    overflow: hidden;
}}
.side-title::after {{
    content: '';
    position: absolute;
    top: 0; left: -60%;
    width: 40%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
    animation: shine 3.2s ease-in-out infinite;
}}
@keyframes shine {{
    0% {{ left: -60%; }}
    60%, 100% {{ left: 140%; }}
}}

.pill {{
    background: {th['card']};
    border: 1px solid {th['card_border']};
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 10px;
    font-size: 0.85rem;
    color: {th['text']};
    backdrop-filter: blur(10px);
}}
.bar {{
    height: 8px;
    border-radius: 6px;
    background: rgba(148, 163, 184, 0.2);
    margin-top: 10px;
    overflow: hidden;
    position: relative;
}}
.bar > div {{
    height: 100%;
    background: linear-gradient(90deg, #00C2FF, #8B5CF6, #FF2E63);
    transition: width 1s cubic-bezier(0.34, 1.56, 0.64, 1);
    box-shadow: 0 0 14px rgba(255, 46, 99, 0.6);
}}

.hero {{
    padding: 32px 38px;
    border-radius: 24px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
    background: linear-gradient(120deg, {th['hero_from']}, {th['hero_mid']} 45%, {th['hero_to']});
    box-shadow: 0 18px 55px rgba(255, 46, 99, 0.4);
}}
.hero::before {{
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 65%; height: 200%;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.16) 0%, transparent 60%);
}}
.hero h1 {{
    color: #fff;
    margin: 0;
    font-size: 2.35rem;
    font-weight: 900;
    letter-spacing: -0.6px;
    position: relative;
}}
.hero p {{
    color: #FFE4EC;
    margin: 8px 0 14px;
    font-size: 1.02rem;
    position: relative;
    line-height: 1.55;
}}
.badge {{
    display: inline-block;
    padding: 5px 14px;
    margin: 0 6px 6px 0;
    border-radius: 20px;
    font-size: 0.74rem;
    background: rgba(255, 255, 255, 0.18);
    color: #fff;
    border: 1px solid rgba(255, 255, 255, 0.32);
    font-weight: 600;
    backdrop-filter: blur(8px);
    transition: all 0.25s ease;
}}
.badge:hover {{
    background: rgba(255, 255, 255, 0.35);
    transform: translateY(-2px);
}}

.ticker {{
    overflow: hidden;
    white-space: nowrap;
    border-radius: 12px;
    padding: 12px 0;
    margin-bottom: 18px;
    background: {th['ticker_bg']};
    border: 1px solid {th['ticker_border']};
    backdrop-filter: blur(8px);
}}
.track {{
    display: inline-block;
    padding-left: 100%;
    animation: scroll 65s linear infinite;
    color: {th['ticker_text']};
    font-weight: 600;
    font-size: 0.88rem;
}}
.track span {{ margin-right: 80px; }}
@keyframes scroll {{
    to {{ transform: translateX(-100%); }}
}}

.glass {{
    background: {th['card']};
    border: 1px solid {th['card_border']};
    border-radius: 18px;
    padding: 20px 22px;
    transition: all 0.35s ease;
    height: 100%;
    backdrop-filter: blur(12px);
}}
.glass:hover {{
    transform: translateY(-5px);
    border-color: {th['card_border_glow']};
    box-shadow: 0 14px 40px rgba(255, 46, 99, 0.25);
}}
.glass h4 {{
    margin: 0 0 8px;
    color: {th['text']};
    font-size: 1.05rem;
    font-weight: 800;
}}
.glass p {{
    margin: 0;
    color: {th['sub']};
    font-size: 0.88rem;
    line-height: 1.55;
}}

.metric-card-box {{
    background: {th['card']};
    border: 1px solid {th['card_border']};
    border-radius: 16px;
    padding: 16px 18px;
    transition: all 0.3s ease;
    height: 100%;
    backdrop-filter: blur(10px);
}}
.metric-card-box:hover {{
    transform: translateY(-4px);
    border-color: #FF2E63;
    box-shadow: 0 10px 30px rgba(255, 46, 99, 0.25);
}}
.metric-card-box .lbl {{
    font-size: 0.72rem;
    color: {th['sub']};
    text-transform: uppercase;
    letter-spacing: 1.4px;
    font-weight: 800;
}}
.metric-card-box .val {{
    font-size: 1.75rem;
    font-weight: 900;
    color: {th['text']};
    margin-top: 4px;
    letter-spacing: -0.5px;
}}
.metric-card-box .sub {{
    font-size: 0.76rem;
    color: {th['sub']};
    margin-top: 3px;
    font-weight: 500;
}}

.kpi-container {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin-bottom: 18px;
}}
.kpi-card {{
    background: linear-gradient(135deg, rgba(255,46,99,0.92) 0%, rgba(127,16,50,0.95) 100%);
    color: #fff;
    border-radius: 18px;
    padding: 18px 20px;
    transition: all 0.35s ease;
    box-shadow: 0 10px 28px rgba(255,46,99,0.35);
    border: 1px solid rgba(255,255,255,0.18);
}}
.kpi-card:nth-child(2) {{
    background: linear-gradient(135deg, rgba(0,194,255,0.92) 0%, rgba(15,23,42,0.95) 100%);
    box-shadow: 0 10px 28px rgba(0,194,255,0.35);
}}
.kpi-card:nth-child(3) {{
    background: linear-gradient(135deg, rgba(139,92,246,0.92) 0%, rgba(30,27,75,0.95) 100%);
    box-shadow: 0 10px 28px rgba(139,92,246,0.35);
}}
.kpi-card:nth-child(4) {{
    background: linear-gradient(135deg, rgba(255,184,0,0.92) 0%, rgba(120,53,15,0.95) 100%);
    box-shadow: 0 10px 28px rgba(255,184,0,0.35);
}}
.kpi-card:nth-child(5) {{
    background: linear-gradient(135deg, rgba(16,229,176,0.92) 0%, rgba(6,78,59,0.95) 100%);
    box-shadow: 0 10px 28px rgba(16,229,176,0.35);
}}
.kpi-card:nth-child(6) {{
    background: linear-gradient(135deg, rgba(255,110,199,0.92) 0%, rgba(131,24,67,0.95) 100%);
    box-shadow: 0 10px 28px rgba(255,110,199,0.35);
}}
.kpi-card:hover {{
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 14px 38px rgba(255,46,99,0.5);
}}
.kpi-card .k-lbl {{
    font-size: 0.72rem;
    color: rgba(255, 255, 255, 0.92);
    letter-spacing: 1.3px;
    text-transform: uppercase;
    font-weight: 800;
}}
.kpi-card .k-val {{
    font-size: 1.7rem;
    font-weight: 900;
    margin-top: 5px;
    letter-spacing: -0.5px;
    color: #fff;
}}
.kpi-card .k-sub {{
    font-size: 0.74rem;
    color: rgba(255, 255, 255, 0.82);
    margin-top: 4px;
}}

.pred-banner {{
    padding: 24px 28px;
    border-radius: 20px;
    background: linear-gradient(120deg, #00C2FF 0%, #8B5CF6 50%, #FF2E63 100%);
    color: #fff;
    text-align: center;
    box-shadow: 0 14px 44px rgba(139, 92, 246, 0.45);
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}}
.pred-banner h2 {{
    margin: 0;
    font-size: 1.4rem;
    font-weight: 900;
    color: #fff;
}}
.pred-banner .amt {{
    font-size: 3rem;
    font-weight: 900;
    margin: 6px 0;
    letter-spacing: -1.5px;
    color: #fff;
}}
.pred-banner .amt-pkr {{
    font-size: 1.4rem;
    font-weight: 800;
    color: rgba(255, 255, 255, 0.94);
}}
.pred-banner .meta {{
    font-size: 0.88rem;
    color: rgba(255, 255, 255, 0.9);
    font-weight: 600;
    margin-top: 4px;
}}

.formula-box {{
    padding: 12px 18px;
    border-radius: 12px;
    background: {th['accent']};
    border: 1px dashed #FF2E63;
    font-family: 'JetBrains Mono', monospace;
    color: {th['text']};
    font-size: 0.88rem;
    font-weight: 600;
    margin-top: 8px;
    overflow-x: auto;
}}

div[data-testid="stPlotlyChart"] {{
    background: {th['card']};
    border: 1px solid {th['card_border']};
    border-radius: 18px;
    padding: 8px;
    transition: all 0.35s ease;
}}
div[data-testid="stPlotlyChart"]:hover {{
    transform: translateY(-4px);
    border-color: {th['card_border_glow']};
    box-shadow: 0 16px 45px rgba(255, 46, 99, 0.25);
}}

button[data-baseweb="tab"] {{
    font-size: 0.95rem;
    font-weight: 700;
    color: {th['text']} !important;
    transition: all 0.25s ease;
    padding: 12px 18px;
}}
button[data-baseweb="tab"]:hover {{
    color: #FF2E63 !important;
    transform: translateY(-1px);
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: #FF2E63 !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{
    background: #FF2E63 !important;
    height: 3px;
}}

.stMarkdown, .stText, label, p, span, div {{
    color: {th['text']};
}}

.badge-tag {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 6px;
}}
.badge-tag-red {{ background: rgba(255,46,99,0.18); color: #FF2E63; border: 1px solid #FF2E63; }}
.badge-tag-blue {{ background: rgba(0,194,255,0.18); color: #00C2FF; border: 1px solid #00C2FF; }}
.badge-tag-gold {{ background: rgba(255,184,0,0.18); color: #FFB800; border: 1px solid #FFB800; }}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PLOTLY STYLING HELPERS
# ============================================================
def style_fig(fig, height=420, animate=False):
    fig.update_layout(
        template=th["plotly_tpl"],
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=th["plot_bg"],
        font=dict(color=th["text"], family="Inter", size=12),
        margin=dict(l=14, r=14, t=55, b=14),
        legend_title_text="",
        title_font=dict(size=15, color=th["text"], family="Inter"),
        xaxis=dict(
            gridcolor=th["grid"],
            zerolinecolor=th["grid"],
            tickfont=dict(color=th["text"]),
            title_font=dict(color=th["text"]),
        ),
        yaxis=dict(
            gridcolor=th["grid"],
            zerolinecolor=th["grid"],
            tickfont=dict(color=th["text"]),
            title_font=dict(color=th["text"]),
        ),
        legend=dict(font=dict(color=th["text"])),
        hoverlabel=dict(bgcolor=RED, font=dict(color="#fff", family="Inter", size=12)),
    )
    if animate:
        fig.update_layout(
            transition=dict(duration=600, easing="cubic-in-out")
        )
    return fig


def add_play_button(fig, duration=900, label="Play Animation"):
    """Single clean custom play button on animated Plotly charts."""
    fig.update_layout(updatemenus=[])
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=1.16,
                bgcolor="rgba(255,46,99,.92)",
                bordercolor="#fff",
                font=dict(color="#fff", family="Inter", size=11),
                buttons=[
                    dict(
                        label=f"▶ {label}",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=duration, redraw=True),
                                transition=dict(duration=400, easing="cubic-in-out"),
                                fromcurrent=True,
                            ),
                        ],
                    )
                ],
            )
        ]
    )
    return fig


def add_best_fit_line(fig, x, y, color="#FF2E63", name="Regression Trendline"):
    """Adds polyfit linear regression line + 95% confidence band matching RegressionTypeProj.py regplot."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.ptp(x) == 0:
        return fig
    m, c = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = m * xs + c
    yhat = m * x + c
    resid = y - yhat
    std = np.std(resid)
    upper = ys + 1.96 * std
    lower = ys - 1.96 * std

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([xs, xs[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(255, 46, 99, 0.12)",
            line=dict(color="rgba(255, 46, 99, 0)"),
            hoverinfo="skip",
            showlegend=True,
            name="95% Confidence Band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=name,
            line=dict(color=color, width=3.2),
            hovertemplate="x=%{x:.1f}<br>y=%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_annotation(
        x=0.98,
        y=0.04,
        xref="paper",
        yref="paper",
        text=f"y = {m:,.1f}x + {c:,.0f}",
        showarrow=False,
        font=dict(color=color, family="JetBrains Mono", size=11),
        bgcolor="rgba(255, 46, 99, 0.15)",
        bordercolor=color,
        borderwidth=1,
        borderpad=5,
    )
    return fig


def metric_card(label, value, sub=""):
    return (
        f'<div class="metric-card-box"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def glass_card(title, text):
    return f'<div class="glass"><h4>{title}</h4><p>{text}</p></div>'


# ============================================================
# DATA & MODEL ENGINE (EXACT PARITY WITH RegressionTypeProj.py)
# ============================================================
@st.cache_data
def load_data():
    file_path = os.path.join(
        os.path.dirname(__file__), "insurance_80_percent_train.xlsx"
    )
    df = pd.read_excel(file_path)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 30, 40, 50, 64],
        labels=["18-30", "31-40", "41-50", "51-64"],
    )
    df["bmi_class"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )
    return df


@st.cache_resource
def train_linear_model(raw_df):
    """
    Trains the exact linear regression model specified in RegressionTypeProj.py:
    1. Categorical one-hot encoding with drop_first=True
    2. Feature Engineering:
       - bmi_smoker = bmi * smoker_yes
       - bmi_obese  = (bmi >= 30).astype(int)
    3. Target log1p(charges)
    4. 5-Fold Cross Validation on full (X, y)
    5. Train/Test Split (80/20, random_state=42)
    """
    # 3. Encode Categorical Features
    encoded = pd.get_dummies(
        raw_df, columns=["sex", "smoker", "region"], drop_first=True, dtype=int
    )

    # 4. Feature Engineering
    encoded["bmi_smoker"] = encoded["bmi"] * encoded["smoker_yes"]
    encoded["bmi_obese"] = (encoded["bmi"] >= 30).astype(int)

    # 5. Separate Features and Target
    X = encoded.drop("charges", axis=1)
    y = np.log1p(encoded["charges"])
    feature_cols = list(X.columns)

    # 6. Cross Validation (on full X, y exactly as in RegressionTypeProj.py)
    model = LinearRegression()
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    # 7. Train/Test Split (20% test held out, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)

    # Exact RegressionTypeProj.py log-space metrics
    test_r2_log = float(r2_score(y_test, y_pred_log))
    test_mse_log = float(mean_squared_error(y_test, y_pred_log))

    # Raw USD Charges (expm1 inverse transform)
    act_usd = np.expm1(y_test.values)
    pred_usd = np.expm1(y_pred_log)
    test_r2_usd = float(r2_score(act_usd, pred_usd))
    test_rmse_usd = float(np.sqrt(mean_squared_error(act_usd, pred_usd)))
    test_mae_usd = float(mean_absolute_error(act_usd, pred_usd))

    # Log residual standard deviation (for actuarial prediction intervals)
    log_resid_std = float(np.std(y_test.values - y_pred_log))

    metrics = dict(
        cv_scores=cv_scores,
        cv_mean=float(cv_scores.mean()),
        cv_std=float(cv_scores.std()),
        test_r2_log=test_r2_log,
        test_mse_log=test_mse_log,
        test_r2_usd=test_r2_usd,
        test_rmse_usd=test_rmse_usd,
        test_mae_usd=test_mae_usd,
        log_resid_std=log_resid_std,
        n_train=len(X_train),
        n_test=len(X_test),
        intercept=float(model.intercept_),
        coefficients=dict(zip(feature_cols, model.coef_)),
    )

    return model, feature_cols, metrics, X_test, y_test, act_usd, pred_usd


def predict_patient(model, cols, age, bmi, children, sex, smoker, region, log_resid_std=0.389):
    """
    Predicts charges for an individual patient with exact feature alignment
    and calculates 95% actuarial prediction intervals.
    """
    row = {c: 0 for c in cols}
    row["age"] = float(age)
    row["bmi"] = float(bmi)
    row["children"] = int(children)

    # One-hot flags (drop_first dropped female, smoker_no, region_northeast)
    if sex.lower() == "male" and "sex_male" in row:
        row["sex_male"] = 1
    if smoker.lower() == "yes" and "smoker_yes" in row:
        row["smoker_yes"] = 1

    region_key = f"region_{region.lower()}"
    if region_key in row:
        row[region_key] = 1

    # Exact Feature Engineering
    row["bmi_smoker"] = float(bmi) * row.get("smoker_yes", 0)
    row["bmi_obese"] = 1 if bmi >= 30 else 0

    input_df = pd.DataFrame([row])[cols]
    pred_log = float(model.predict(input_df)[0])
    pred_usd = float(np.expm1(pred_log))

    # 95% prediction interval in log space, converted back to raw dollars
    lower_log = pred_log - 1.96 * log_resid_std
    upper_log = pred_log + 1.96 * log_resid_std
    lower_usd = float(max(0.0, np.expm1(lower_log)))
    upper_usd = float(np.expm1(upper_log))

    return pred_usd, lower_usd, upper_usd, pred_log


def predict_batch_dataframe(model, cols, input_df):
    """
    Safely and robustly predicts on custom batch datasets.
    Handles column casing, alias variations, and computes interactions properly.
    """
    d = input_df.copy()
    # Normalize column names to lowercase
    col_map = {c: c.strip().lower() for c in d.columns}
    d = d.rename(columns=col_map)

    # Standardize values
    if "smoker" in d.columns:
        d["smoker"] = d["smoker"].astype(str).str.strip().str.lower()
    if "sex" in d.columns:
        d["sex"] = d["sex"].astype(str).str.strip().str.lower()
    if "region" in d.columns:
        d["region"] = d["region"].astype(str).str.strip().str.lower()

    # Pre-compute smoker flag for interaction
    smoker_flag = (d["smoker"] == "yes").astype(int) if "smoker" in d.columns else pd.Series(0, index=d.index)

    # One-hot encode
    encoded = pd.get_dummies(d, columns=["sex", "smoker", "region"], drop_first=True, dtype=int)

    # Add feature engineered columns
    encoded["bmi_smoker"] = d["bmi"] * smoker_flag
    encoded["bmi_obese"] = (d["bmi"] >= 30).astype(int)

    # Align with model training columns
    for c in cols:
        if c not in encoded.columns:
            encoded[c] = 0

    X_aligned = encoded[cols]
    pred_log = model.predict(X_aligned)
    pred_usd = np.expm1(pred_log)
    return np.maximum(pred_usd, 0.0)


# ============================================================
# LOAD DATASET & TRAIN MODEL
# ============================================================
data = load_data()
raw_subset = data[["age", "sex", "bmi", "children", "smoker", "region", "charges"]].copy()
model, cols, M, X_test, y_test, act_usd, pred_usd = train_linear_model(raw_subset)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
sb = st.sidebar
sb.markdown('<div class="side-title">FILTER CONTROL CENTER</div>', unsafe_allow_html=True)

# Settings & Visual Modes
c_pref1, c_pref2 = sb.columns(2)
theme_choice = c_pref1.radio(
    "Theme",
    ["Dark", "Light"],
    index=0 if st.session_state.theme == "dark" else 1,
    horizontal=True,
    key="sb_theme_select",
)
new_theme = "dark" if theme_choice == "Dark" else "light"
if new_theme != st.session_state.theme:
    st.session_state.theme = new_theme
    st.rerun()

curr_choice = c_pref2.selectbox(
    "Currency",
    ["Dual", "USD ($)", "PKR (Rs)"],
    index=0 if st.session_state.currency_mode == "dual" else (1 if st.session_state.currency_mode == "usd" else 2),
    key="sb_curr_select",
)
new_curr = "dual" if curr_choice == "Dual" else ("usd" if "USD" in curr_choice else "pkr")
if new_curr != st.session_state.currency_mode:
    st.session_state.currency_mode = new_curr
    st.rerun()

curr_mode = st.session_state.currency_mode
sb.markdown("---")

# Data Filters
with sb.expander("👤 Demographics", expanded=True):
    smoker_opts = sorted(data["smoker"].unique())
    sel_smoker = st.multiselect("Smoker Status", smoker_opts, default=smoker_opts, key="f_smoker")

    sex_opts = sorted(data["sex"].unique())
    sel_sex = st.multiselect("Sex", sex_opts, default=sex_opts, key="f_sex")

with sb.expander("📍 Geography & Region", expanded=False):
    region_opts = sorted(data["region"].unique())
    sel_region = st.multiselect("Region", region_opts, default=region_opts, key="f_region")

with sb.expander("👨‍👩‍👧 Family Dependents", expanded=False):
    kids_all = [int(k) for k in sorted(data["children"].unique())]
    sel_kids = st.multiselect("Children", kids_all, default=kids_all, key="f_kids")

with sb.expander("⚖️ Age & BMI Range", expanded=False):
    a0, a1 = int(data["age"].min()), int(data["age"].max())
    sel_age = st.slider("Age Range", a0, a1, (a0, a1), key="f_age")

    b0, b1 = float(np.floor(data["bmi"].min())), float(np.ceil(data["bmi"].max()))
    sel_bmi = st.slider("BMI Range", b0, b1, (b0, b1), step=0.5, key="f_bmi")

# Apply Cohort Filters
df = data[
    data["smoker"].isin(sel_smoker)
    & data["sex"].isin(sel_sex)
    & data["region"].isin(sel_region)
    & data["children"].isin(sel_kids)
    & data["age"].between(*sel_age)
    & data["bmi"].between(*sel_bmi)
]

pct_active = len(df) / len(data) * 100
sb.markdown(
    f"""
    <div class="pill">
        Cohort: <b>{len(df):,}</b> of {len(data):,} records ({pct_active:.0f}%)
        <div class="bar"><div style="width:{pct_active:.0f}%"></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
sb.caption("Note: Filters isolate cohort visuals. ML model is trained on the full benchmark dataset.")

# ============================================================
# HERO BANNER
# ============================================================
st.markdown(
    f"""
<div class="hero">
  <h1>Insurance Cost Intelligence Studio</h1>
  <p>An enterprise machine learning workspace powered by <b>Linear Regression</b> with interaction engineering,
     log-transformation diagnostics, and 5-fold cross-validation verification.</p>
  <span class="badge">scikit-learn LinearRegression</span>
  <span class="badge">5-Fold CV: {M['cv_mean']:.3f}</span>
  <span class="badge">Test R²: {M['test_r2_log']:.3f}</span>
  <span class="badge">Test MSE: {M['test_mse_log']:.4f}</span>
  <span class="badge">Log-Target Transform: log1p</span>
  <span class="badge">Dual Currency: USD & PKR</span>
</div>
""",
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("⚠️ No records match the current filter selection. Please expand your sidebar filters.")
    st.stop()

# ============================================================
# DYNAMIC INSIGHT TICKER
# ============================================================
s_yes = df[df["smoker"] == "yes"]["charges"]
s_no = df[df["smoker"] == "no"]["charges"]

ticker_items = [
    f"📊 {len(df):,} Patients in Active Cohort",
    f"💵 Cohort Avg Charge: {fmt_money(df['charges'].mean(), curr_mode)}",
]
if len(s_yes) and len(s_no):
    ticker_items.append(f"🚬 Smokers Pay {s_yes.mean() / s_no.mean():.1f}x More than Non-Smokers on Average")
if len(df):
    top_region = df.groupby("region")["charges"].mean().idxmax().title()
    obese_pct = (df["bmi"] >= 30).mean() * 100
    ticker_items.append(f"📍 Highest Cost Region: {top_region}")
    ticker_items.append(f"⚖️ Clinical Obesity Rate: {obese_pct:.1f}%")
ticker_items.append(f"🎯 Model Test R²: {M['test_r2_log']:.4f} (Log) | {M['test_r2_usd']:.4f} (USD)")
ticker_items.append(f"💱 Exchange Rate: 1 USD = PKR {USD_TO_PKR:.2f}")

ticker_clean = list(dict.fromkeys(ticker_items))
st.markdown(
    f"""
<div class="ticker">
  <div class="track">
    {''.join(f'<span>{t}</span>' for t in ticker_clean * 2)}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# RESPONSIVE KPI CARDS (CLEAN & ERROR-FREE)
# ============================================================
avg_charge_val = float(df["charges"].mean())
max_charge_val = float(df["charges"].max())
median_charge_val = float(df["charges"].median())
avg_bmi_val = float(df["bmi"].mean())
smoker_pct_val = float((df["smoker"] == "yes").mean() * 100)

kpi_html = f"""
<div class="kpi-container">
  <div class="kpi-card">
    <div class="k-lbl">Active Cohort</div>
    <div class="k-val">{len(df):,}</div>
    <div class="k-sub">{pct_active:.0f}% of full training pool</div>
  </div>
  <div class="kpi-card">
    <div class="k-lbl">Average Cost</div>
    <div class="k-val">{fmt_usd(avg_charge_val) if curr_mode == 'usd' else (fmt_pkr(avg_charge_val) if curr_mode == 'pkr' else fmt_usd(avg_charge_val))}</div>
    <div class="k-sub">{fmt_pkr(avg_charge_val) if curr_mode == 'dual' else 'Per individual'}</div>
  </div>
  <div class="kpi-card">
    <div class="k-lbl">Median Cost</div>
    <div class="k-val">{fmt_usd(median_charge_val) if curr_mode == 'usd' else (fmt_pkr(median_charge_val) if curr_mode == 'pkr' else fmt_usd(median_charge_val))}</div>
    <div class="k-sub">{fmt_pkr(median_charge_val) if curr_mode == 'dual' else '50th percentile'}</div>
  </div>
  <div class="kpi-card">
    <div class="k-lbl">Highest Observed Cost</div>
    <div class="k-val">{fmt_usd(max_charge_val) if curr_mode == 'usd' else (fmt_pkr(max_charge_val) if curr_mode == 'pkr' else fmt_usd(max_charge_val))}</div>
    <div class="k-sub">{fmt_pkr(max_charge_val) if curr_mode == 'dual' else 'Observed maximum'}</div>
  </div>
  <div class="kpi-card">
    <div class="k-lbl">Average BMI</div>
    <div class="k-val">{avg_bmi_val:.1f}</div>
    <div class="k-sub">{'Overweight range' if avg_bmi_val >= 25 else 'Normal range'}</div>
  </div>
  <div class="kpi-card">
    <div class="k-lbl">Smoker Prevalence</div>
    <div class="k-val">{smoker_pct_val:.1f}%</div>
    <div class="k-sub">{int(round(smoker_pct_val * len(df) / 100)):,} smokers active</div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)
st.caption(
    "💡 Tip: Highest Observed Cost indicates the peak individual annual policy charge within this filtered cohort. "
    "Averages and medians highlight the true central distribution."
)

# ============================================================
# TABS NAVIGATION
# ============================================================
tab_overview, tab_eda, tab_story, tab_metrics, tab_predict, tab_batch = st.tabs([
    "📊 Cohort Overview",
    "🔬 EDA & Regression Lab",
    "✨ Animated Storytelling",
    "🎯 Model Diagnostics",
    "🔮 Predict Charges",
    "🧪 Test Custom Data",
])

# ============================================================
# TAB 1: COHORT OVERVIEW
# ============================================================
with tab_overview:
    # Top 3 High-Level Storytelling Cards
    ov_c1, ov_c2, ov_c3 = st.columns(3)
    with ov_c1:
        if len(s_yes) and len(s_no):
            ov_c1.markdown(
                glass_card(
                    "🔥 Smoking: The #1 Cost Catalyst",
                    f"Smokers average <b>{fmt_money(s_yes.mean(), curr_mode)}</b> compared to <b>{fmt_money(s_no.mean(), curr_mode)}</b> for non-smokers. "
                    f"Smoking amplifies expected medical expenses by an astronomical <b>{s_yes.mean() / s_no.mean():.1f}x</b>.",
                ),
                unsafe_allow_html=True,
            )
        else:
            ov_c1.markdown(glass_card("Smoking Cohort", "Single smoking status selected in filters."), unsafe_allow_html=True)

    with ov_c2:
        ob = df[df["bmi"] >= 30]["charges"]
        nob = df[df["bmi"] < 30]["charges"]
        if len(ob) and len(nob):
            ov_c2.markdown(
                glass_card(
                    "⚖️ The Obesity Threshold Jump",
                    f"Patients with BMI ≥ 30 average <b>{fmt_money(ob.mean(), curr_mode)}</b> versus <b>{fmt_money(nob.mean(), curr_mode)}</b> for BMI &lt; 30. "
                    "This threshold jump is captured in our model by the engineered <code>bmi_obese</code> flag.",
                ),
                unsafe_allow_html=True,
            )
        else:
            ov_c2.markdown(glass_card("BMI Threshold", "Selected cohort contains uniform BMI range."), unsafe_allow_html=True)

    with ov_c3:
        ag = df.groupby("age_group", observed=True)["charges"].mean()
        if len(ag) > 1:
            ov_c3.markdown(
                glass_card(
                    "📈 Age Compounding Trend",
                    f"Age group <b>{ag.index[0]}</b> averages <b>{fmt_money(ag.iloc[0], curr_mode)}</b>, steadily compounding to "
                    f"<b>{fmt_money(ag.iloc[-1], curr_mode)}</b> for <b>{ag.index[-1]}</b>.",
                ),
                unsafe_allow_html=True,
            )
        else:
            ov_c3.markdown(glass_card("Age Trajectory", "Single age group filtered."), unsafe_allow_html=True)

    st.write("")

    # Visual Row 1: Regional Age Progression & Smoker Ratio
    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        anim_df = (
            df.groupby(["region", "age_group"], observed=True)["charges"]
            .mean()
            .reset_index()
        )
        if not anim_df.empty:
            f1 = px.bar(
                anim_df,
                x="age_group",
                y="charges",
                animation_frame="region",
                color="charges",
                range_y=[0, anim_df["charges"].max() * 1.2],
                color_continuous_scale=th["scale"],
                title="Average Charges by Age Group (Animated across Regions)",
                labels={"age_group": "Age Group", "charges": "Average Charges (USD)"},
            )
            f1.update_coloraxes(showscale=False)
            f1.update_traces(marker_line_width=0)
            f1.layout.updatemenus = ()
            add_play_button(f1, 950, "Play Regional Walkthrough")
            c1.plotly_chart(style_fig(f1, 410), use_container_width=True)

    with c2:
        share = df["smoker"].value_counts().reset_index()
        share.columns = ["smoker", "count"]
        f2 = px.pie(
            share,
            names="smoker",
            values="count",
            hole=0.6,
            color="smoker",
            color_discrete_map=SMOKER_MAP,
            title="Smoker Cohort Breakdown",
        )
        f2.update_traces(
            textinfo="percent+label",
            pull=[0.06] * len(share),
            marker=dict(line=dict(color="#fff", width=2.5)),
            textfont=dict(size=13, color="#fff", family="Inter"),
        )
        c2.plotly_chart(style_fig(f2, 410), use_container_width=True)

    # Visual Row 2: Distribution & Feature Correlations
    c3, c4 = st.columns(2)
    with c3:
        f3 = px.histogram(
            df,
            x="charges",
            color="smoker",
            nbins=35,
            barmode="overlay",
            opacity=0.78,
            color_discrete_map=SMOKER_MAP,
            title="Distribution of Charges (Smokers vs Non-Smokers)",
            labels={"charges": "Annual Charges (USD)", "smoker": "Smoker"},
        )
        f3.update_traces(marker_line_width=0)
        c3.plotly_chart(style_fig(f3, 400), use_container_width=True)

    with c4:
        encoded_full = pd.get_dummies(
            df[["age", "bmi", "children", "sex", "smoker", "region"]],
            drop_first=True,
            dtype=int,
        )
        encoded_full["bmi_smoker"] = encoded_full["bmi"] * encoded_full.get("smoker_yes", 0)
        encoded_full["bmi_obese"] = (df["bmi"] >= 30).astype(int)
        corr_series = encoded_full.corrwith(df["charges"]).dropna().sort_values()

        f4 = px.bar(
            x=corr_series.values,
            y=corr_series.index,
            orientation="h",
            color=corr_series.values,
            color_continuous_scale=th["scale"],
            title="Feature Correlation with Insurance Charges",
            labels={"x": "Pearson Correlation Coefficient", "y": "Feature"},
        )
        f4.update_coloraxes(showscale=False)
        f4.update_traces(marker_line_width=0)
        f4.add_vline(x=0, line_dash="solid", line_color=th["sub"], line_width=1)
        c4.plotly_chart(style_fig(f4, 400), use_container_width=True)

# ============================================================
# TAB 2: EDA & LINEAR REGRESSION LAB (EXACT RegressionTypeProj.py PARITY)
# ============================================================
with tab_eda:
    st.markdown("### 🔬 Exploratory Data Analysis & Regression Workflow")
    st.markdown(
        """
        <div class="formula-box">
            Mathematical Model: log(charges + 1) = β₀ + β₁(age) + β₂(bmi) + β₃(children) + β₄(sex_male) + β₅(smoker_yes) + β₆(bmi_smoker) + β₇(bmi_obese) + β(region)
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "📌 This tab provides an interactive implementation of **Steps 2, 3, and 4** from your "
        "<code>RegressionTypeProj.py</code> script: visualizing numerical features with regression trendlines, "
        "categorical strip plots with diamond average markers, and feature interactions."
    )

    # ----------------------------------------------------
    # NUMERICAL FEATURES: regplot PARITY
    # ----------------------------------------------------
    st.markdown("#### 1. Numerical Features vs Charges (`sns.regplot` Parity)")
    num_col1, num_col2, num_col3 = st.columns(3)

    with num_col1:
        f_age = px.scatter(
            df,
            x="age",
            y="charges",
            color="smoker",
            opacity=0.68,
            color_discrete_map=SMOKER_MAP,
            title="age vs charges (Scatter + Regression Line)",
            hover_data=["bmi", "children", "region"],
        )
        f_age.update_traces(marker=dict(size=8, line=dict(width=0.8, color="#fff")))
        add_best_fit_line(f_age, df["age"].to_numpy(), df["charges"].to_numpy(), color=RED, name="Best-Fit Line")
        f_age.update_layout(yaxis_title="Charges (USD)")
        num_col1.plotly_chart(style_fig(f_age, 420), use_container_width=True)

    with num_col2:
        f_bmi = px.scatter(
            df,
            x="bmi",
            y="charges",
            color="smoker",
            opacity=0.68,
            color_discrete_map=SMOKER_MAP,
            title="bmi vs charges (Threshold at BMI 30)",
            hover_data=["age", "children", "region"],
        )
        f_bmi.update_traces(marker=dict(size=8, line=dict(width=0.8, color="#fff")))
        f_bmi.add_vline(
            x=30,
            line_dash="dash",
            line_color=AMBER,
            line_width=2.5,
            annotation_text="BMI 30 (Obese)",
            annotation_font_color=AMBER,
            annotation_position="top left",
        )
        add_best_fit_line(f_bmi, df["bmi"].to_numpy(), df["charges"].to_numpy(), color=RED, name="Best-Fit Line")
        f_bmi.update_layout(yaxis_title="Charges (USD)")
        num_col2.plotly_chart(style_fig(f_bmi, 420), use_container_width=True)

    with num_col3:
        # In RegressionTypeProj.py: x_jitter=0.1 if col == 'children' else 0
        df_jitter = df.copy()
        df_jitter["children_jitter"] = df_jitter["children"] + np.random.uniform(-0.12, 0.12, size=len(df_jitter))
        f_kids = px.scatter(
            df_jitter,
            x="children_jitter",
            y="charges",
            color="smoker",
            opacity=0.65,
            color_discrete_map=SMOKER_MAP,
            title="children vs charges (With Jitter 0.1)",
            hover_data=["children", "age", "bmi"],
        )
        f_kids.update_traces(marker=dict(size=8, line=dict(width=0.8, color="#fff")))
        add_best_fit_line(f_kids, df["children"].to_numpy(), df["charges"].to_numpy(), color=RED, name="Best-Fit Line")
        f_kids.update_layout(xaxis_title="children (jittered for visibility)", yaxis_title="Charges (USD)")
        num_col3.plotly_chart(style_fig(f_kids, 420), use_container_width=True)

    # ----------------------------------------------------
    # CATEGORICAL FEATURES: stripplot + pointplot PARITY
    # ----------------------------------------------------
    st.markdown("#### 2. Categorical Features vs Charges (`sns.stripplot` + `sns.pointplot` Parity)")
    cat_view_mode = st.radio(
        "Display Mode",
        [
            "Strip Plot + Average Diamond Point (RegressionTypeProj.py Exact Match)",
            "Violin & Box Plot Distributions",
        ],
        horizontal=True,
        key="cat_view_toggle",
    )

    cat_col1, cat_col2, cat_col3 = st.columns(3)

    if "Strip" in cat_view_mode:
        # Exact reproduction of RegressionTypeProj.py: stripplot with jitter + black diamond pointplot
        for col_name, col_widget, palette_colors in [
            ("sex", cat_col1, [PINK, SKY]),
            ("smoker", cat_col2, [SKY, RED]),
            ("region", cat_col3, [PURPLE, SKY, RED, MINT]),
        ]:
            with col_widget:
                fig_strip = go.Figure()
                # 1. Stripplot points (category observations with jitter)
                categories = sorted(df[col_name].unique())
                for idx, cat_val in enumerate(categories):
                    cat_subset = df[df[col_name] == cat_val]
                    jitter_x = idx + np.random.uniform(-0.22, 0.22, size=len(cat_subset))
                    pt_color = palette_colors[idx % len(palette_colors)]
                    fig_strip.add_trace(
                        go.Scatter(
                            x=jitter_x,
                            y=cat_subset["charges"],
                            mode="markers",
                            name=str(cat_val),
                            marker=dict(size=6, color=pt_color, opacity=0.6),
                            hovertemplate=f"{col_name}: {cat_val}<br>Charges: $%{{y:,.0f}}<extra></extra>",
                        )
                    )

                # 2. Pointplot (Black Diamond for average charges per category)
                mean_charges = df.groupby(col_name)["charges"].mean().loc[categories]
                fig_strip.add_trace(
                    go.Scatter(
                        x=list(range(len(categories))),
                        y=mean_charges.values,
                        mode="markers",
                        name="Average (◆)",
                        marker=dict(symbol="diamond", size=13, color="#000000", line=dict(color="#ffffff", width=2)),
                        hovertemplate=f"{col_name}: %{{x}}<br>Mean Charge: $%{{y:,.0f}}<extra></extra>",
                    )
                )

                fig_strip.update_layout(
                    title=f"{col_name} vs charges (Strip + Mean Diamond)",
                    xaxis=dict(tickmode="array", tickvals=list(range(len(categories))), ticktext=categories, title=col_name),
                    yaxis=dict(title="Charges (USD)"),
                    showlegend=False,
                )
                col_widget.plotly_chart(style_fig(fig_strip, 420), use_container_width=True)
    else:
        with cat_col1:
            f_v1 = px.violin(
                df, x="sex", y="charges", color="sex", box=True, points="all",
                color_discrete_sequence=[PINK, SKY], title="sex vs charges (Violin Distribution)",
            )
            f_v1.update_traces(marker=dict(opacity=0.45, size=4), line_color="#fff", meanline_visible=True)
            cat_col1.plotly_chart(style_fig(f_v1, 420), use_container_width=True)

        with cat_col2:
            f_v2 = px.violin(
                df, x="smoker", y="charges", color="smoker", box=True, points="all",
                color_discrete_map=SMOKER_MAP, title="smoker vs charges (Violin Distribution)",
            )
            f_v2.update_traces(marker=dict(opacity=0.45, size=4), line_color="#fff", meanline_visible=True)
            cat_col2.plotly_chart(style_fig(f_v2, 420), use_container_width=True)

        with cat_col3:
            f_v3 = px.box(
                df, x="region", y="charges", color="region", points="outliers",
                color_discrete_map=REGION_MAP, title="region vs charges (Boxplot Outliers)",
            )
            cat_col3.plotly_chart(style_fig(f_v3, 420), use_container_width=True)

    # ----------------------------------------------------
    # FEATURE ENGINEERING SHOWCASE (Step 4 of script)
    # ----------------------------------------------------
    st.markdown("#### 3. Feature Engineering Deep Dive: Interaction & Obesity Jump")
    fe_col1, fe_col2 = st.columns(2)

    with fe_col1:
        st.markdown(
            """
            <div class="glass">
                <h4>Interaction: <code>bmi_smoker = bmi * smoker_yes</code></h4>
                <p>In classical linear regression, adding BMI alone assumes that each BMI point increases medical costs equally for everyone.
                However, medical reality shows that <b>excess weight combined with smoking produces an explosive multiplicative effect</b>.
                Notice below how the regression slope for smokers is dramatically steeper than for non-smokers!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        f_inter = px.scatter(
            df,
            x="bmi",
            y="charges",
            color="smoker",
            color_discrete_map=SMOKER_MAP,
            opacity=0.75,
            trendline="ols",
            title="BMI vs Charges by Smoker Status (Dual Slope Demonstration)",
            labels={"bmi": "Body Mass Index (BMI)", "charges": "Charges (USD)"},
        )
        f_inter.update_traces(marker=dict(size=8, line=dict(width=0.8, color="#fff")))
        fe_col1.plotly_chart(style_fig(f_inter, 420), use_container_width=True)

    with fe_col2:
        st.markdown(
            """
            <div class="glass">
                <h4>Threshold Flag: <code>bmi_obese = (bmi >= 30).astype(int)</code></h4>
                <p>The World Health Organization defines BMI ≥ 30 as clinical obesity.
                Our binary indicator captures the discrete jump in chronic health liabilities (cardiovascular risk, diabetes)
                that occurs once a patient crosses this medical threshold.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        df_obese_comp = df.copy()
        df_obese_comp["obesity_status"] = np.where(df_obese_comp["bmi"] >= 30, "Obese (BMI ≥ 30)", "Non-Obese (BMI < 30)")
        f_obese_bar = px.box(
            df_obese_comp,
            x="obesity_status",
            y="charges",
            color="smoker",
            color_discrete_map=SMOKER_MAP,
            points="outliers",
            title="Charges Distribution Across Obesity Threshold",
            labels={"obesity_status": "Clinical Classification", "charges": "Charges (USD)"},
        )
        fe_col2.plotly_chart(style_fig(f_obese_bar, 420), use_container_width=True)

# ============================================================
# TAB 3: ANIMATED STORYTELLING & 3D ANALYTICS
# ============================================================
with tab_story:
    st.markdown("### ✨ Dynamic Data Storytelling & Multidimensional Space")
    st.caption("Experience how patient features evolve across life stages and multidimensional feature space.")

    st_col1, st_col2 = st.columns(2)

    with st_col1:
        d3_anim = df.sort_values("age_group")
        f_bubble = px.scatter(
            d3_anim,
            x="bmi",
            y="charges",
            color="smoker",
            size="age",
            animation_frame="age_group",
            opacity=0.85,
            color_discrete_map=SMOKER_MAP,
            size_max=22,
            range_x=[df["bmi"].min() - 1, df["bmi"].max() + 1],
            range_y=[0, df["charges"].max() * 1.15],
            title="BMI vs Charges (Bubble Size = Age, Animated by Age Group)",
            labels={"bmi": "BMI", "charges": "Charges (USD)"},
        )
        f_bubble.update_traces(marker=dict(line=dict(width=1.5, color="#fff")))
        f_bubble.add_vline(x=30, line_dash="dash", line_color=AMBER, opacity=0.85, annotation_text="BMI 30")
        f_bubble.layout.updatemenus = ()
        add_play_button(f_bubble, 900, "Play Age Progression")
        st_col1.plotly_chart(style_fig(f_bubble, 480), use_container_width=True)

    with st_col2:
        g_cum = df.groupby(["age", "smoker"])["charges"].mean().reset_index()
        subs = {s: g_cum[g_cum["smoker"] == s].sort_values("age") for s in g_cum["smoker"].unique()}
        max_len = max(len(v) for v in subs.values()) if subs else 0

        def make_age_traces(frame_idx):
            traces = []
            for s_val, sub_df in subs.items():
                traces.append(
                    go.Scatter(
                        x=sub_df["age"][:frame_idx],
                        y=sub_df["charges"][:frame_idx],
                        mode="lines+markers",
                        name=f"Smoker: {s_val}",
                        line=dict(color=SMOKER_MAP.get(s_val, SKY), width=3.5),
                        marker=dict(size=8, line=dict(width=1.8, color="#fff")),
                    )
                )
            return traces

        if max_len >= 2:
            f_cum = go.Figure(
                data=make_age_traces(2),
                frames=[go.Frame(data=make_age_traces(i), name=str(i)) for i in range(2, max_len + 1)],
            )
            f_cum.update_layout(
                title="Progressive Life Trajectory: Age vs Charges",
                xaxis=dict(range=[df["age"].min() - 1, df["age"].max() + 1], title="Patient Age"),
                yaxis=dict(range=[0, g_cum["charges"].max() * 1.15], title="Average Charges (USD)"),
            )
            add_play_button(f_cum, 120, "Trace Trajectory")
            st_col2.plotly_chart(style_fig(f_cum, 480), use_container_width=True)

    # 3D Rotating Feature Space
    st.markdown("#### 3D Interactive Feature Topology (`Age` × `BMI` × `Charges`)")
    sample_size = min(len(df), 1200)
    df_sample_3d = df.sample(sample_size, random_state=42)

    f_3d = go.Figure(
        go.Scatter3d(
            x=df_sample_3d["age"],
            y=df_sample_3d["bmi"],
            z=df_sample_3d["charges"],
            mode="markers",
            marker=dict(
                size=4.5,
                color=df_sample_3d["charges"],
                colorscale=th["scale"],
                opacity=0.88,
                line=dict(width=0.5, color="#fff"),
            ),
            hovertemplate="Age: %{x}<br>BMI: %{y:.1f}<br>Charges: $%{z:,.0f}<extra></extra>",
        )
    )

    f_3d.frames = [
        go.Frame(layout=dict(scene_camera=dict(eye=dict(x=2.2 * np.cos(t), y=2.2 * np.sin(t), z=0.9))))
        for t in np.linspace(0, 2 * np.pi, 60)
    ]
    f_3d.update_layout(
        title="3D Feature Manifold (Auto-Rotating Camera)",
        scene=dict(
            xaxis_title="Age",
            yaxis_title="BMI",
            zaxis_title="Charges (USD)",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=th["grid"], tickfont=dict(color=th["text"])),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=th["grid"], tickfont=dict(color=th["text"])),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=th["grid"], tickfont=dict(color=th["text"])),
        ),
    )
    add_play_button(f_3d, 70, "Rotate 360°")
    st.plotly_chart(style_fig(f_3d, 580), use_container_width=True)

# ============================================================
# TAB 4: MODEL PERFORMANCE & MATHEMATICAL RIGOR
# ============================================================
with tab_metrics:
    st.markdown("### 🎯 Model Performance & Validation Rigor")
    st.markdown(
        """
        <div class="glass" style="margin-bottom:16px;">
            <h4>✅ Exact Parity with <code>RegressionTypeProj.py</code></h4>
            <p>Our model strictly reproduces every step of your main script: 5-fold cross validation on <code>(X, y)</code>,
            80/20 train-test split (<code>random_state=42</code>), log1p target transformation, and identical feature engineering.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Exact Benchmarks Display
    bench_col1, bench_col2, bench_col3, bench_col4, bench_col5, bench_col6 = st.columns(6)
    bench_col1.markdown(
        metric_card("5-Fold Mean R²", f"{M['cv_mean']:.4f}", f"± {M['cv_std']:.4f} std"),
        unsafe_allow_html=True,
    )
    bench_col2.markdown(
        metric_card("Test R² (Log)", f"{M['test_r2_log']:.4f}", "Exact script match"),
        unsafe_allow_html=True,
    )
    bench_col3.markdown(
        metric_card("Test MSE (Log)", f"{M['test_mse_log']:.4f}", "Exact script match"),
        unsafe_allow_html=True,
    )
    bench_col4.markdown(
        metric_card("Test R² (USD)", f"{M['test_r2_usd']:.4f}", "Raw dollar space"),
        unsafe_allow_html=True,
    )
    bench_col5.markdown(
        metric_card("Test RMSE", fmt_usd(M["test_rmse_usd"]), fmt_pkr(M["test_rmse_usd"])),
        unsafe_allow_html=True,
    )
    bench_col6.markdown(
        metric_card("Test MAE", fmt_usd(M["test_mae_usd"]), fmt_pkr(M["test_mae_usd"])),
        unsafe_allow_html=True,
    )

    st.write("")

    # Validation Visuals: 5-Fold Scores & Actual vs Predicted
    m_v1, m_v2 = st.columns(2)

    with m_v1:
        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(M["cv_scores"]))],
            "R2_Score": M["cv_scores"],
        })
        f_cv = px.bar(
            cv_df,
            x="Fold",
            y="R2_Score",
            text="R2_Score",
            color="R2_Score",
            color_continuous_scale=th["scale"],
            title="5-Fold Cross-Validation Consistency (Scoring: R²)",
            range_y=[0.65, 0.9],
        )
        f_cv.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            marker_line_width=0,
        )
        f_cv.add_hline(
            y=M["cv_mean"],
            line_dash="dash",
            line_color=AMBER,
            line_width=2.5,
            annotation_text=f"Mean R²: {M['cv_mean']:.3f}",
            annotation_font_color=AMBER,
        )
        f_cv.update_coloraxes(showscale=False)
        m_v1.plotly_chart(style_fig(f_cv, 440), use_container_width=True)

    with m_v2:
        f_avp = go.Figure()
        f_avp.add_trace(
            go.Scatter(
                x=act_usd,
                y=pred_usd,
                mode="markers",
                name="Test Data Points",
                marker=dict(color=RED, size=8, opacity=0.75, line=dict(width=1, color="#fff")),
                hovertemplate="Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>",
            )
        )
        min_v = min(act_usd.min(), pred_usd.min())
        max_v = max(act_usd.max(), pred_usd.max())
        f_avp.add_trace(
            go.Scatter(
                x=[min_v, max_v],
                y=[min_v, max_v],
                mode="lines",
                name="Perfect Line (y = x)",
                line=dict(color=AMBER, dash="dash", width=3),
            )
        )
        f_avp.update_layout(
            title=f"Actual vs Predicted Charges (Test R² = {M['test_r2_usd']:.3f})",
            xaxis_title="Actual Insurance Charges (USD)",
            yaxis_title="Predicted Insurance Charges (USD)",
        )
        m_v2.plotly_chart(style_fig(f_avp, 440), use_container_width=True)

    # Residuals Diagnostics: Error Distribution & Homoscedasticity Check
    r_v1, r_v2 = st.columns(2)

    residuals = act_usd - pred_usd
    with r_v1:
        f_res = px.histogram(
            x=residuals,
            nbins=40,
            color_discrete_sequence=[PURPLE],
            title="Residual Error Distribution (Actual - Predicted USD)",
            labels={"x": "Residual Error ($ USD)"},
        )
        f_res.add_vline(x=0, line_dash="dash", line_color=AMBER, line_width=2.5)
        f_res.update_traces(marker_line_width=0)
        r_v1.plotly_chart(style_fig(f_res, 380), use_container_width=True)

    with r_v2:
        f_homo = go.Figure()
        f_homo.add_trace(
            go.Scatter(
                x=pred_usd,
                y=residuals,
                mode="markers",
                name="Residuals",
                marker=dict(color=SKY, size=7, opacity=0.7, line=dict(width=0.8, color="#fff")),
                hovertemplate="Fitted: $%{x:,.0f}<br>Residual: $%{y:,.0f}<extra></extra>",
            )
        )
        f_homo.add_hline(y=0, line_dash="solid", line_color=RED, line_width=2)
        f_homo.update_layout(
            title="Residuals vs Fitted Values (Homoscedasticity Diagnostic)",
            xaxis_title="Fitted / Predicted Charges ($ USD)",
            yaxis_title="Residual Error ($ USD)",
        )
        r_v2.plotly_chart(style_fig(f_homo, 380), use_container_width=True)

    # Model Equation & Coefficients Table
    st.markdown("#### 📐 Model Coefficients & Feature Multipliers")
    coef_series = pd.Series(model.coef_, index=cols).sort_values()

    c_left, c_right = st.columns([1.2, 0.8])
    with c_left:
        f_coef = px.bar(
            x=coef_series.values,
            y=coef_series.index,
            orientation="h",
            color=coef_series.values > 0,
            color_discrete_map={True: RED, False: SKY},
            title="Learned Coefficients (Impact on log1p Charges)",
            labels={"x": "Coefficient Weight (β)", "y": "Feature"},
        )
        f_coef.update_layout(showlegend=False)
        f_coef.update_traces(marker_line_width=0)
        f_coef.add_vline(x=0, line_dash="solid", line_color=th["sub"], line_width=1)
        c_left.plotly_chart(style_fig(f_coef, 420), use_container_width=True)

    with c_right:
        coef_table_data = []
        for feature, coef_val in zip(cols, model.coef_):
            pct_effect = (np.exp(coef_val) - 1.0) * 100
            coef_table_data.append({
                "Feature": feature,
                "Weight (β)": f"{coef_val:+.4f}",
                "Multiplier Effect": f"{pct_effect:+.2f}%",
            })
        coef_df = pd.DataFrame(coef_table_data)
        st.markdown(f"**Model Intercept (β₀):** `{model.intercept_:.4f}`")
        st.caption("Because target y = log(charges + 1), each coefficient β means a 1-unit increase in X multiplies baseline cost by exp(β).")
        st.dataframe(coef_df, hide_index=True, use_container_width=True)

# ============================================================
# TAB 5: PREDICT CHARGES (ACTUARIAL ESTIMATOR)
# ============================================================
with tab_predict:
    st.markdown("### 🔮 Individual Insurance Cost Estimator")
    st.caption("Adjust patient profile parameters below for real-time actuarial estimation with 95% confidence intervals.")

    p_in_col, p_out_col = st.columns([1, 1.25])

    with p_in_col:
        st.markdown("##### 👤 Patient Parameters")
        in_age = st.slider("Patient Age", 18, 64, 38, key="pred_age")
        in_bmi = st.slider("Body Mass Index (BMI)", 15.0, 52.0, 28.5, step=0.1, key="pred_bmi")
        in_kids = st.select_slider("Number of Children", options=[0, 1, 2, 3, 4, 5], value=1, key="pred_kids")

        c_sub1, c_sub2 = st.columns(2)
        in_sex = c_sub1.selectbox("Biological Sex", sorted(data["sex"].unique()), key="pred_sex")
        in_smoker = c_sub2.radio("Smoking Status", sorted(data["smoker"].unique()), horizontal=True, key="pred_smoker")
        in_region = st.selectbox("Geographic Region", sorted(data["region"].unique()), key="pred_region")

    # Run Prediction for this profile
    pred_charge, pred_low, pred_high, pred_log_val = predict_patient(
        model, cols, in_age, in_bmi, in_kids, in_sex, in_smoker, in_region, M["log_resid_std"]
    )

    with p_out_col:
        st.markdown(
            f"""
        <div class="pred-banner">
            <h2>Estimated Annual Insurance Charge</h2>
            <div class="amt">{fmt_usd(pred_charge)}</div>
            <div class="amt-pkr">{fmt_pkr(pred_charge)}</div>
            <div class="meta">
                95% Prediction Range: {fmt_usd(pred_low)} – {fmt_usd(pred_high)}<br>
                PKR Range: {fmt_pkr(pred_low)} – {fmt_pkr(pred_high)}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Actuarial Risk Gauge
        max_gauge = max(float(data["charges"].max()), pred_charge * 1.15)
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pred_charge,
                number={"prefix": "$", "font": {"size": 40, "color": RED, "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, max_gauge], "tickcolor": th["text"]},
                    "bar": {"color": RED, "thickness": 0.3},
                    "steps": [
                        {"range": [0, 10000], "color": "rgba(0,194,255,0.25)"},
                        {"range": [10000, 25000], "color": "rgba(255,184,0,0.25)"},
                        {"range": [25000, max_gauge], "color": "rgba(255,46,99,0.25)"},
                    ],
                    "threshold": {"line": {"color": "#ffffff", "width": 4}, "thickness": 0.85, "value": pred_charge},
                },
            )
        )
        p_out_col.plotly_chart(style_fig(gauge_fig, 260), use_container_width=True)

    # ----------------------------------------------------
    # COUNTERFACTUAL WHAT-IF SCENARIOS
    # ----------------------------------------------------
    st.markdown("#### 🔄 What-If Counterfactual Simulations")
    wf1, wf2, wf3 = st.columns(3)

    # Scenario 1: Smoking Impact
    pred_no_smoke, _, _, _ = predict_patient(model, cols, in_age, in_bmi, in_kids, in_sex, "no", in_region, M["log_resid_std"])
    pred_smoke, _, _, _ = predict_patient(model, cols, in_age, in_bmi, in_kids, in_sex, "yes", in_region, M["log_resid_std"])
    smoke_diff = pred_smoke - pred_no_smoke

    with wf1:
        st.markdown(
            glass_card(
                "🚬 Smoking Cessation Impact",
                f"If this individual is a smoker, quitting will save approximately <b>{fmt_money(smoke_diff, curr_mode)}</b> annually! "
                f"<br>Non-Smoker: {fmt_usd(pred_no_smoke)} | Smoker: {fmt_usd(pred_smoke)}",
            ),
            unsafe_allow_html=True,
        )

    # Scenario 2: BMI Optimization
    pred_normal_bmi, _, _, _ = predict_patient(model, cols, in_age, 24.5, in_kids, in_sex, in_smoker, in_region, M["log_resid_std"])
    bmi_savings = max(0.0, pred_charge - pred_normal_bmi)

    with wf2:
        st.markdown(
            glass_card(
                "⚖️ Weight Optimization Impact",
                f"Reaching a normal BMI of 24.5 would save approximately <b>{fmt_money(bmi_savings, curr_mode)}</b> per year. "
                f"<br>Current BMI ({in_bmi}): {fmt_usd(pred_charge)} | Normal (24.5): {fmt_usd(pred_normal_bmi)}",
            ),
            unsafe_allow_html=True,
        )

    # Scenario 3: 5-Year Aging Progression
    pred_future_age, _, _, _ = predict_patient(model, cols, min(64, in_age + 5), in_bmi, in_kids, in_sex, in_smoker, in_region, M["log_resid_std"])
    age_increase = pred_future_age - pred_charge

    with wf3:
        st.markdown(
            glass_card(
                "⏳ 5-Year Aging Progression",
                f"In 5 years (at age {min(64, in_age + 5)}), projected annual charges will increase by <b>{fmt_money(age_increase, curr_mode)}</b>. "
                f"<br>Age {in_age}: {fmt_usd(pred_charge)} | Age {min(64, in_age + 5)}: {fmt_usd(pred_future_age)}",
            ),
            unsafe_allow_html=True,
        )

# ============================================================
# TAB 6: TEST CUSTOM DATA & BATCH INTELLIGENCE
# ============================================================
with tab_batch:
    st.markdown("### 🧪 Test Custom Data & Batch Intelligence")
    st.caption(
        "Upload a CSV or Excel file containing patient profiles to generate batch predictions. "
        "If a `charges` column is present, the dashboard automatically evaluates model accuracy metrics."
    )

    # Sample template download
    sample_test_data = pd.DataFrame([
        {"age": 19, "sex": "female", "bmi": 27.9, "children": 0, "smoker": "yes", "region": "southwest", "charges": 16884.92},
        {"age": 33, "sex": "male", "bmi": 22.7, "children": 0, "smoker": "no", "region": "northwest", "charges": 4449.46},
        {"age": 47, "sex": "female", "bmi": 32.0, "children": 1, "smoker": "no", "region": "southeast", "charges": 8556.91},
        {"age": 52, "sex": "male", "bmi": 34.4, "children": 3, "smoker": "yes", "region": "northeast", "charges": 42112.24},
        {"age": 28, "sex": "female", "bmi": 24.3, "children": 2, "smoker": "no", "region": "northwest", "charges": 5152.13},
    ])
    sample_csv = sample_test_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Sample 5-Patient Test CSV",
        data=sample_csv,
        file_name="sample_insurance_test.csv",
        mime="text/csv",
        help="Use this sample file to immediately test batch prediction functionality.",
    )

    st.write("")
    uploaded_file = st.file_uploader(
        "Upload Custom Patient Dataset (.csv or .xlsx)",
        type=["csv", "xlsx", "xls"],
        key="batch_file_uploader",
    )

    # ----------------------------------------------------
    # SINGLE PATIENT QUICK FORM
    # ----------------------------------------------------
    with st.expander("📝 Or Test a Single Patient Manually", expanded=False):
        with st.form("manual_single_form", clear_on_submit=False):
            f_c1, f_c2, f_c3, f_c4 = st.columns(4)
            m_age = f_c1.number_input("Age", 18, 100, 36, step=1)
            m_sex = f_c2.selectbox("Sex", ["male", "female"])
            m_bmi = f_c3.number_input("BMI", 12.0, 60.0, 28.0, step=0.1)
            m_kids = f_c4.number_input("Children", 0, 10, 1, step=1)

            f_c5, f_c6, f_c7 = st.columns(3)
            m_smoker = f_c5.selectbox("Smoker", ["no", "yes"])
            m_region = f_c6.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
            m_actual = f_c7.number_input("Actual Charges (USD) [Optional]", 0.0, 200000.0, 0.0, step=100.0)

            f_submit = st.form_submit_button("Predict for Patient", use_container_width=True)

        if f_submit:
            single_pred, s_low, s_high, _ = predict_patient(
                model, cols, m_age, m_bmi, m_kids, m_sex, m_smoker, m_region, M["log_resid_std"]
            )
            st.markdown(
                f"""
            <div class="pred-banner">
                <h2>Prediction Result</h2>
                <div class="amt">{fmt_usd(single_pred)}</div>
                <div class="amt-pkr">{fmt_pkr(single_pred)}</div>
                <div class="meta">
                    Profile: {m_age} yrs | {m_sex} | BMI {m_bmi:.1f} | {m_kids} kids | {m_smoker} smoker | {m_region}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            if m_actual > 0:
                diff = m_actual - single_pred
                pct_err = (diff / m_actual) * 100
                score = max(0.0, 100.0 - abs(pct_err))
                st.markdown(
                    f"**Actual Charge:** {fmt_usd(m_actual)} | **Predicted:** {fmt_usd(single_pred)} | "
                    f"**Difference:** {fmt_usd(diff)} ({pct_err:+.1f}%) | **Accuracy:** `{score:.1f}%`"
                )
                if abs(pct_err) < 15:
                    st.success("🎯 High Accuracy Prediction (within 15% error margin).")
                elif abs(pct_err) < 30:
                    st.info("ℹ️ Good Prediction (within normal actuarial variance).")
                else:
                    st.warning("⚠️ High Variance: Patient may have outlier medical conditions not captured in basic demographics.")

    # ----------------------------------------------------
    # BATCH FILE PROCESSOR
    # ----------------------------------------------------
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                user_df = pd.read_csv(uploaded_file)
            else:
                user_df = pd.read_excel(uploaded_file)

            st.success(f"✅ Successfully loaded `{len(user_df):,}` rows with columns: `{list(user_df.columns)}`")

            # Check required columns
            norm_cols = [c.strip().lower() for c in user_df.columns]
            required_cols = ["age", "sex", "bmi", "children", "smoker", "region"]
            missing_cols = [rc for rc in required_cols if rc not in norm_cols]

            if missing_cols:
                st.error(f"❌ Missing required columns in uploaded dataset: `{missing_cols}`. Please ensure your file includes: {required_cols}")
            else:
                with st.spinner("Executing Linear Regression inference..."):
                    batch_preds = predict_batch_dataframe(model, cols, user_df)

                results_df = user_df.copy()
                results_df["predicted_charges_usd"] = np.round(batch_preds, 2)
                results_df["predicted_charges_pkr"] = np.round(batch_preds * USD_TO_PKR, 2)

                st.markdown("#### 📋 Batch Predictions Preview")
                st.dataframe(results_df.head(100), use_container_width=True)

                # Accuracy evaluation if actual 'charges' exist
                charges_col = [c for c in user_df.columns if c.strip().lower() == "charges"]
                if charges_col:
                    actual_charges = user_df[charges_col[0]].values
                    test_r2 = r2_score(actual_charges, batch_preds)
                    test_rmse = np.sqrt(mean_squared_error(actual_charges, batch_preds))
                    test_mae = mean_absolute_error(actual_charges, batch_preds)
                    denom = np.where(actual_charges == 0, 1.0, actual_charges)
                    mape = np.mean(np.abs((actual_charges - batch_preds) / denom)) * 100
                    accuracy_pct = max(0.0, 100.0 - mape)

                    st.markdown("#### 🎯 Model Evaluation on Uploaded Ground Truth")
                    b1, b2, b3, b4, b5 = st.columns(5)
                    b1.markdown(metric_card("Batch R² Score", f"{test_r2:.4f}", "Variance explained"), unsafe_allow_html=True)
                    b2.markdown(metric_card("Batch RMSE", fmt_usd(test_rmse), fmt_pkr(test_rmse)), unsafe_allow_html=True)
                    b3.markdown(metric_card("Batch MAE", fmt_usd(test_mae), fmt_pkr(test_mae)), unsafe_allow_html=True)
                    b4.markdown(metric_card("MAPE", f"{mape:.2f}%", "Mean Abs % Error"), unsafe_allow_html=True)
                    b5.markdown(metric_card("Accuracy Score", f"{accuracy_pct:.2f}%", "100 - MAPE"), unsafe_allow_html=True)

                    st.write("")

                    # Evaluation Charts
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        f_batch_scat = go.Figure()
                        f_batch_scat.add_trace(
                            go.Scatter(
                                x=actual_charges,
                                y=batch_preds,
                                mode="markers",
                                name="Uploaded Patients",
                                marker=dict(color=RED, size=8, opacity=0.75, line=dict(width=0.8, color="#fff")),
                                hovertemplate="Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>",
                            )
                        )
                        b_min = min(actual_charges.min(), batch_preds.min())
                        b_max = max(actual_charges.max(), batch_preds.max())
                        f_batch_scat.add_trace(
                            go.Scatter(
                                x=[b_min, b_max],
                                y=[b_min, b_max],
                                mode="lines",
                                name="Ideal (y = x)",
                                line=dict(color=AMBER, dash="dash", width=3),
                            )
                        )
                        f_batch_scat.update_layout(
                            title=f"Actual vs Predicted on Uploaded Data (R² = {test_r2:.3f})",
                            xaxis_title="Actual Charges (USD)",
                            yaxis_title="Predicted Charges (USD)",
                        )
                        bc1.plotly_chart(style_fig(f_batch_scat, 420), use_container_width=True)

                    with bc2:
                        batch_res = actual_charges - batch_preds
                        f_b_res = px.histogram(
                            x=batch_res,
                            nbins=35,
                            color_discrete_sequence=[PURPLE],
                            title="Residual Error Distribution on Uploaded Data",
                            labels={"x": "Prediction Error ($ USD)"},
                        )
                        f_b_res.add_vline(x=0, line_dash="dash", line_color=AMBER, line_width=2.5)
                        f_b_res.update_traces(marker_line_width=0)
                        bc2.plotly_chart(style_fig(f_b_res, 420), use_container_width=True)
                else:
                    st.info("ℹ️ Uploaded dataset does not contain a `charges` column. Displaying predicted charges.")

                # Export Results
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Download All Predictions as CSV",
                    data=csv_bytes,
                    file_name="insurance_predictions_output.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as err:
            st.error(f"⚠️ Error processing file: {err}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center; color:{th['sub']}; font-size:0.82rem; padding:12px 0;">
        🏥 <b>Insurance Cost Intelligence Studio</b> — Built with Python, Pandas, Plotly, Streamlit, and scikit-learn Linear Regression.
    </div>
    """,
    unsafe_allow_html=True,
)