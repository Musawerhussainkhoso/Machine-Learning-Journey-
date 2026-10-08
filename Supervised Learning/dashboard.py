import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Insurance Cost Intelligence | Linear Regression Studio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CURRENCY CONVERSION
# Original dataset (IBM Medical Cost Personal) is in USD.
# Exchange rate below is a documented reference value.
# Source: State Bank of Pakistan interbank rate, Oct 2024.
# Update this constant if you need a different reference date.
# ============================================================
USD_TO_PKR = 278.50


def to_pkr(usd):
    return usd * USD_TO_PKR


def usd_str(usd):
    return f"${usd:,.0f}"


def pkr_str(usd):
    return f"PKR {to_pkr(usd):,.0f}"


def dual_str(usd):
    """Return both currencies in one readable string."""
    return f"{usd_str(usd)}  |  {pkr_str(usd)}"


# ============================================================
# THEME SYSTEM
# ============================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

RED = '#FF2E63'
SKY = '#00C2FF'
AMBER = '#FFB800'
PURPLE = '#8B5CF6'
MINT = '#10E5B0'
PINK = '#FF6EC7'

SMOKER_MAP = {'yes': RED, 'no': SKY}
REGION_MAP = {'northeast': PURPLE, 'northwest': SKY, 'southeast': RED, 'southwest': MINT}


def T():
    if st.session_state.theme == "dark":
        return dict(
            bg="radial-gradient(circle at 15% 15%, #1a0a2e 0%, #070a18 40%, #0d1230 70%, #070a18 100%)",
            card="rgba(255,255,255,.06)",
            card_border="rgba(255,255,255,.14)",
            text="#F1F5F9",
            sub="#94a3b8",
            hero_from="#FF2E63", hero_mid="#7F1032", hero_to="#1e1b4b",
            sidebar="linear-gradient(180deg,#14091c 0%,#0a0d22 100%)",
            sidebar_border="rgba(255,46,99,.4)",
            plotly_tpl="plotly_dark",
            ticker_bg="rgba(255,255,255,.06)",
            ticker_text="#ffd6e0",
            scale=['#0f172a', '#1e1b4b', '#4c1d95', '#8B5CF6', '#FF2E63', '#FFB800'],
            accent="rgba(255,46,99,.15)",
            grid="rgba(148,163,184,.15)",
            plot_bg="rgba(255,255,255,.02)",
        )
    return dict(
        bg="radial-gradient(circle at 15% 15%, #ffe8f0 0%, #f0f4ff 40%, #eef6ff 70%, #fff5f8 100%)",
        card="#ffffff",
        card_border="rgba(255,46,99,.22)",
        text="#0f172a",
        sub="#475569",
        hero_from="#FF2E63", hero_mid="#B3123C", hero_to="#6366f1",
        sidebar="#ffffff",
        sidebar_border="rgba(255,46,99,.25)",
        plotly_tpl="plotly_white",
        ticker_bg="#ffffff",
        ticker_text="#7F1032",
        scale=['#dbeafe', '#93c5fd', '#6366f1', '#8B5CF6', '#FF2E63', '#FFB800'],
        accent="rgba(255,46,99,.1)",
        grid="rgba(100,116,139,.2)",
        plot_bg="#f8fafc",
    )


th = T()

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter','Segoe UI',sans-serif; }}
.stApp {{ background: {th['bg']}; background-attachment: fixed; }}
.stApp::before, .stApp::after {{
    content:''; position:fixed; border-radius:50%; filter:blur(120px);
    z-index:0; pointer-events:none; opacity:.25;
}}
.stApp::before {{ width:480px; height:480px; background:#FF2E63; top:-140px; left:-120px; }}
.stApp::after  {{ width:520px; height:520px; background:#00C2FF; bottom:-180px; right:-140px; }}

#MainMenu, footer {{ visibility:hidden; height:0; }}

button[data-testid="collapsedControl"] {{
    background: linear-gradient(135deg, #FF2E63, #7F1032) !important;
    color: #fff !important;
    border-radius: 0 12px 12px 0 !important;
    box-shadow: 0 8px 24px rgba(255,46,99,.55) !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
    height: 60px !important;
    width: 40px !important;
    top: 45% !important;
}}
button[data-testid="collapsedControl"]:hover {{
    width: 50px !important;
    box-shadow: 0 0 34px rgba(255,46,99,.9) !important;
}}
button[data-testid="collapsedControl"] svg {{ fill: #fff !important; }}

header[data-testid="stHeader"] {{
    background: transparent !important;
    visibility: visible !important;
    height: 2.5rem !important;
}}

.block-container {{ padding-top:1rem; max-width:1500px; }}

section[data-testid="stSidebar"] {{
    background: {th['sidebar']} !important;
    border-right:1px solid {th['sidebar_border']};
    box-shadow: 6px 0 40px rgba(255,46,99,.1);
}}
section[data-testid="stSidebar"] * {{ color:{th['text']} !important; }}
.side-title {{
    font-weight:900; letter-spacing:2.5px; font-size:.9rem; padding:14px 18px;
    border-radius:14px; background:linear-gradient(120deg,#FF2E63,#7F1032);
    color:#fff !important; margin-bottom:14px; text-align:center;
    box-shadow:0 8px 28px rgba(255,46,99,.5);
    position: relative; overflow: hidden;
}}
.side-title::after {{
    content: ''; position: absolute; top:0; left:-60%; width:40%; height:100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.35), transparent);
    animation: shine 3s ease-in-out infinite;
}}
@keyframes shine {{ 0% {{ left:-60%; }} 60%,100% {{ left:140%; }} }}

section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
    background:{th['card']}; border:1px solid {th['card_border']};
    border-radius:12px; transition:.35s ease;
}}
section[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {{
    border-color:#FF2E63; box-shadow:0 0 22px rgba(255,46,99,.45);
}}
span[data-baseweb="tag"] {{
    background:linear-gradient(120deg,#FF2E63,#B3123C) !important;
    border-radius:9px !important;
    box-shadow: 0 3px 10px rgba(255,46,99,.4);
}}
span[data-baseweb="tag"] * {{ color:#fff !important; font-weight:600; }}
div[data-baseweb="slider"] div[role="slider"] {{
    background:#FF2E63 !important;
    box-shadow:0 0 18px #FF2E63, 0 0 6px #fff;
}}

.pill {{
    background:{th['card']}; border:1px solid {th['card_border']};
    border-radius:14px; padding:14px 16px; margin-top:10px;
    font-size:.85rem;
}}
.bar {{
    height:10px; border-radius:8px; background:rgba(128,128,128,.2);
    margin-top:10px; overflow:hidden; position: relative;
}}
.bar > div {{
    height:100%; background:linear-gradient(90deg,#00C2FF,#8B5CF6,#FF2E63);
    transition:width 1.4s cubic-bezier(.34,1.56,.64,1);
    box-shadow: 0 0 16px rgba(255,46,99,.7);
}}

.hero {{
    padding:34px 42px; border-radius:26px; margin-bottom:18px;
    position:relative; overflow:hidden;
    background:linear-gradient(120deg,{th['hero_from']},{th['hero_mid']} 45%,{th['hero_to']});
    box-shadow:0 20px 60px rgba(255,46,99,.5);
}}
.hero::before {{
    content: ''; position: absolute; top:-50%; right:-20%; width: 60%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,.15) 0%, transparent 60%);
}}
.hero h1 {{ color:#fff; margin:0; font-size:2.5rem; font-weight:900; letter-spacing:-.8px; position: relative; }}
.hero p  {{ color:#ffe1e8; margin:10px 0 16px; font-size:1.05rem; position: relative; }}
.badge {{
    display:inline-block; padding:6px 16px; margin:0 8px 6px 0; border-radius:24px;
    font-size:.76rem; background:rgba(255,255,255,.2); color:#fff;
    border:1px solid rgba(255,255,255,.35); font-weight:600;
    transition: .3s ease;
}}
.badge:hover {{ background:rgba(255,255,255,.4); transform:translateY(-2px); }}

.ticker {{
    overflow:hidden; white-space:nowrap; border-radius:14px;
    padding:13px 0; margin-bottom:18px;
    background:{th['ticker_bg']}; border:1px solid {th['card_border']};
}}
.track {{
    display:inline-block; padding-left:100%;
    animation: scroll 55s linear infinite;
    color:{th['ticker_text']}; font-weight:600;
}}
.track span {{ margin-right:90px; }}
@keyframes scroll {{ to {{ transform:translateX(-100%); }} }}

.glass {{
    background:{th['card']}; border:1px solid {th['card_border']};
    border-radius:20px; padding:22px 24px;
    transition:.45s ease; height:100%;
}}
.glass:hover {{
    transform:translateY(-6px);
    border-color:#FF2E63;
    box-shadow:0 16px 46px rgba(255,46,99,.45);
}}
.glass h4 {{ margin:0 0 8px; color:{th['text']}; font-size:1.02rem; font-weight:800; }}
.glass p {{ margin:0; color:{th['sub']}; font-size:.9rem; line-height:1.55; }}

.metric-card {{
    background: {th['card']};
    border: 1px solid {th['card_border']};
    border-radius: 16px; padding: 16px 18px;
    transition: .35s ease;
    height: 100%;
}}
.metric-card:hover {{
    transform: translateY(-4px);
    border-color: #FF2E63;
    box-shadow: 0 12px 34px rgba(255,46,99,.35);
}}
.metric-card .label {{
    font-size: .72rem; color: {th['sub']};
    text-transform: uppercase; letter-spacing: 1.4px; font-weight: 800;
}}
.metric-card .val {{ font-size: 1.85rem; font-weight: 900; color: {th['text']}; margin-top: 4px; }}
.metric-card .sub {{ font-size: .78rem; color: {th['sub']}; margin-top: 3px; }}

.pred-banner {{
    padding: 26px 32px; border-radius: 22px;
    background: linear-gradient(120deg, #00C2FF 0%, #8B5CF6 50%, #FF2E63 100%);
    color: #fff; text-align: center;
    box-shadow: 0 16px 46px rgba(139,92,246,.5);
    margin-bottom: 18px;
}}
.pred-banner h2 {{ margin: 0; font-size: 1.5rem; font-weight: 900; }}
.pred-banner .amt {{ font-size: 3.2rem; font-weight: 900; margin: 8px 0; letter-spacing: -2px; }}
.pred-banner .amt-pkr {{ font-size: 1.5rem; font-weight: 800; opacity: .92; }}
.pred-banner .meta {{ font-size: .95rem; opacity: .92; font-weight: 600; }}

.formula {{
    display: inline-block; padding: 10px 18px; border-radius: 12px;
    background: {th['accent']};
    border: 1px dashed #FF2E63;
    font-family: 'JetBrains Mono', monospace;
    color: {th['text']}; font-size: .92rem; font-weight: 700;
    margin-top: 8px;
}}

div[data-testid="stPlotlyChart"] {{
    background:{th['card']};
    border:1px solid {th['card_border']}; border-radius:20px; padding:10px;
    transition:.45s ease;
}}
div[data-testid="stPlotlyChart"]:hover {{
    transform:translateY(-5px);
    border-color:#FF2E63;
    box-shadow:0 18px 50px rgba(255,46,99,.4);
}}

button[data-baseweb="tab"] {{
    font-size:1rem; font-weight:700; color:{th['text']};
    transition:.3s ease; padding: 12px 18px;
}}
button[data-baseweb="tab"]:hover {{ color:#FF2E63; transform:translateY(-2px); }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:#FF2E63; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:#FF2E63; height: 3px; }}

.stMarkdown, .stText, label, p, span, div {{ color: {th['text']}; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def style_fig(fig, height=420, animate=True):
    fig.update_layout(
        template=th['plotly_tpl'], height=height,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=th['plot_bg'],
        font=dict(color=th['text'], family='Inter', size=12),
        margin=dict(l=10, r=10, t=60, b=10),
        legend_title_text='',
        title_font=dict(size=16, color=th['text'], family='Inter'),
        xaxis=dict(gridcolor=th['grid'], zerolinecolor=th['grid'],
                   tickfont=dict(color=th['text']), title_font=dict(color=th['text'])),
        yaxis=dict(gridcolor=th['grid'], zerolinecolor=th['grid'],
                   tickfont=dict(color=th['text']), title_font=dict(color=th['text'])),
        legend=dict(font=dict(color=th['text'])),
        transition=dict(duration=600, easing='cubic-in-out') if animate else dict(duration=0),
        hoverlabel=dict(bgcolor=RED, font=dict(color='#fff', family='Inter', size=12)),
    )
    return fig
def add_play_button(fig, duration=900, label="Play Animation"):
    """Single clean play button on every animated chart.
    
    FIX: fig.update_layout(updatemenus=[]) must be called FIRST
    to strip the auto-generated Play button that Plotly Express
    creates when animation_frame is used. Without this line, you
    get two Play buttons.
    """
    fig.update_layout(updatemenus=[])   # <-- Line A: clears the auto button
    fig.update_layout(updatemenus=[dict(   # <-- Line B: OVERWRITES Line A
        type="buttons", showactive=False, x=0.02, y=1.15,
        bgcolor='rgba(255,46,99,.9)', bordercolor='#fff',
        font=dict(color='#fff', family='Inter', size=12),
        buttons=[dict(
            label=label, method="animate",
            args=[None, dict(frame=dict(duration=duration, redraw=True),
                             transition=dict(duration=400, easing='cubic-in-out'),
                             fromcurrent=True)]
        )]
    )])
    return fig

def add_best_fit_line(fig, x, y, color='#FFB800', name='Regression Line'):
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
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill='toself', fillcolor='rgba(255,184,0,.12)',
        line=dict(color='rgba(255,184,0,0)'),
        hoverinfo='skip', showlegend=True, name='95% Confidence Band'
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='lines', name=name,
        line=dict(color=color, width=3.5),
        hovertemplate='x=%{x:.2f}<br>y=%{y:,.0f}<extra></extra>'
    ))
    fig.add_annotation(
        x=0.98, y=0.02, xref='paper', yref='paper',
        text=f"y = {m:,.1f}x + {c:,.0f}", showarrow=False,
        font=dict(color=color, family='JetBrains Mono', size=12),
        bgcolor='rgba(255,184,0,.15)', bordercolor=color, borderwidth=1, borderpad=6
    )
    return fig


def show_html(html, height):
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


def glass(title, text):
    return f'<div class="glass"><h4>{title}</h4><p>{text}</p></div>'


def metric_card(label, value, sub=""):
    return (f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="val">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


# ============================================================
# DATA & MODEL
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_excel(os.path.join(os.path.dirname(__file__), 'insurance_80_percent_train.xlsx'))
    df['age_group'] = pd.cut(df['age'], bins=[17, 30, 40, 50, 64],
                             labels=['18-30', '31-40', '41-50', '51-64'])
    df['bmi_class'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100],
                             labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    return df


@st.cache_resource
def train_model(raw):
    d = pd.get_dummies(raw, columns=['sex', 'smoker', 'region'], drop_first=True, dtype=int)
    d['bmi_smoker'] = d['bmi'] * d['smoker_yes']
    d['bmi_obese'] = (d['bmi'] >= 30).astype(int)
    X, y = d.drop('charges', axis=1), np.log1p(d['charges'])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring='r2')
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    act_d, pred_d = np.expm1(y_te.values), np.expm1(pred)
    m = dict(
        cv_mean=cv.mean(), cv_std=cv.std(),
        r2_log=r2_score(y_te, pred), r2_usd=r2_score(act_d, pred_d),
        rmse=float(np.sqrt(mean_squared_error(act_d, pred_d))),
        mae=float(mean_absolute_error(act_d, pred_d)),
        n_train=len(X_tr), n_test=len(X_te)
    )
    return model, list(X.columns), m, act_d, pred_d


def predict_row(model, cols, age, bmi, children, sex, smoker, region):
    row = {c: 0 for c in cols}
    row.update(age=age, bmi=bmi, children=children)
    for k, v in (('sex', sex), ('smoker', smoker), ('region', region)):
        if f'{k}_{v}' in row:
            row[f'{k}_{v}'] = 1
    row['bmi_smoker'] = bmi * row.get('smoker_yes', 0)
    row['bmi_obese'] = int(bmi >= 30)
    return float(np.expm1(model.predict(pd.DataFrame([row])[cols])[0]))


def predict_dataframe(model, cols, df_input):
    d = df_input.copy()
    d = pd.get_dummies(d, columns=['sex', 'smoker', 'region'], drop_first=True, dtype=int)
    for c in cols:
        if c not in d.columns:
            d[c] = 0
    d = d[cols]
    if 'bmi_smoker' in cols:
        d['bmi_smoker'] = df_input['bmi'].values * (df_input['smoker'].astype(str).str.lower() == 'yes').astype(int).values
    if 'bmi_obese' in cols:
        d['bmi_obese'] = (df_input['bmi'].values >= 30).astype(int)
    return np.expm1(model.predict(d))


# ============================================================
# LOAD
# ============================================================
data = load_data()
raw = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']].copy()
model, cols, M, act_d, pred_d = train_model(raw)

# ============================================================
# SIDEBAR
# ============================================================
sb = st.sidebar
sb.markdown('<div class="side-title">FILTER CONTROL CENTER</div>', unsafe_allow_html=True)

theme_choice = sb.radio("Theme", ["Dark", "Light"],
                        index=0 if st.session_state.theme == "dark" else 1,
                        horizontal=True, key="theme_toggle")
new_theme = "dark" if theme_choice == "Dark" else "light"
if new_theme != st.session_state.theme:
    st.session_state.theme = new_theme
    st.rerun()

sb.markdown("---")

with sb.expander("Demographics", expanded=True):
    smoker = st.multiselect("Smoker Status", sorted(data['smoker'].unique()),
                            default=sorted(data['smoker'].unique()), key="f_smoker")
    sex = st.multiselect("Sex", sorted(data['sex'].unique()),
                         default=sorted(data['sex'].unique()), key="f_sex")

with sb.expander("Region", expanded=False):
    region = st.multiselect("Region", sorted(data['region'].unique()),
                            default=sorted(data['region'].unique()), key="f_region")

with sb.expander("Family", expanded=False):
    kids_all = [int(k) for k in sorted(data['children'].unique())]
    kids = st.multiselect("Children", kids_all, default=kids_all, key="f_kids")

with sb.expander("Numeric Ranges", expanded=False):
    a0, a1 = int(data['age'].min()), int(data['age'].max())
    age_range = st.slider("Age Range", a0, a1, (a0, a1), key="f_age")
    b0, b1 = float(np.floor(data['bmi'].min())), float(np.ceil(data['bmi'].max()))
    bmi_range = st.slider("BMI Range", b0, b1, (b0, b1), step=0.5, key="f_bmi")

df = data[data['smoker'].isin(smoker) & data['sex'].isin(sex) & data['region'].isin(region)
          & data['children'].isin(kids) & data['age'].between(*age_range)
          & data['bmi'].between(*bmi_range)]

pct = len(df) / len(data) * 100
sb.markdown(
    f'<div class="pill">Showing <b>{len(df):,}</b> of {len(data):,} records ({pct:.0f}%)'
    f'<div class="bar"><div style="width:{pct:.0f}%"></div></div></div>',
    unsafe_allow_html=True
)
sb.caption("Filters affect charts only. ML model is trained on the full dataset.")

# ============================================================
# HERO
# ============================================================
st.markdown(f"""
<div class="hero">
  <h1>Insurance Cost Intelligence</h1>
  <p>A professional Linear Regression Studio — explore features, visualize relationships,
     and predict insurance charges with real-time accuracy metrics.</p>
  <span class="badge">Python</span><span class="badge">Pandas</span><span class="badge">Plotly</span>
  <span class="badge">Streamlit</span><span class="badge">Linear Regression</span>
  <span class="badge">5-Fold CV</span><span class="badge">USD + PKR</span>
</div>""", unsafe_allow_html=True)

if df.empty:
    st.warning("Filters returned no data. Please adjust the sidebar selections.")
    st.stop()

# ============================================================
# TICKER (de-duplicated)
# ============================================================
tips = [
    f"{len(df):,} patients in current view",
    f"Average charges {dual_str(df['charges'].mean())}",
]
s_yes, s_no = df[df['smoker'] == 'yes']['charges'], df[df['smoker'] == 'no']['charges']
if len(s_yes) and len(s_no):
    tips.append(f"Smokers pay {s_yes.mean() / s_no.mean():.1f}x more than non-smokers")
if len(df):
    tips.append(f"Highest average region: {df.groupby('region')['charges'].mean().idxmax().title()}")
    tips.append(f"{(df['bmi'] >= 30).mean() * 100:.0f}% of patients have BMI 30+")
tips.append(f"Model Test R2 = {M['r2_usd']:.3f}")
tips.append(f"Exchange rate: 1 USD = PKR {USD_TO_PKR:.2f}")

# Remove any accidental duplicates, then duplicate once for seamless scroll loop.
tips = list(dict.fromkeys(tips))
st.markdown('<div class="ticker"><div class="track">'
            + ''.join(f'<span>{t}</span>' for t in tips * 2) + '</div></div>',
            unsafe_allow_html=True)

# ============================================================
# KPI CARDS
# ============================================================
kpis = [
    {"label": "Total Patients", "value": len(df), "prefix": "", "suffix": "", "dec": 0},
    {"label": "Avg Insurance Cost", "value": float(df['charges'].mean()),
     "prefix": "USD", "suffix": "", "dec": 0},
    {"label": "Avg Insurance Cost", "value": float(df['charges'].mean() * USD_TO_PKR),
     "prefix": "PKR ", "suffix": "", "dec": 0},
    {"label": "Highest Insurance Cost", "value": float(df['charges'].max()),
     "prefix": "USD", "suffix": "", "dec": 0},
    {"label": "Highest Insurance Cost", "value": float(df['charges'].max() * USD_TO_PKR),
     "prefix": "PKR ", "suffix": "", "dec": 0},
    {"label": "Avg BMI", "value": float(df['bmi'].mean()), "prefix": "", "suffix": "", "dec": 1},
    {"label": "Smoker Rate", "value": float((df['smoker'] == 'yes').mean() * 100),
     "prefix": "PKR", "suffix": "%", "dec": 1},
]

show_html("""
<style>
 body{margin:0;font-family:'Inter',sans-serif;background:transparent}
 .row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}
 .card{
    background:linear-gradient(135deg,#FF2E63 0%,#7F1032 100%);color:#fff;
    border-radius:20px;padding:18px 20px;position:relative;overflow:hidden;
    transition:.45s ease;
    box-shadow:0 12px 32px rgba(255,46,99,.45);
 }
 .card:nth-child(even){background:linear-gradient(135deg,#00C2FF 0%,#0f172a 100%);box-shadow:0 12px 32px rgba(0,194,255,.4)}
 .card:nth-child(3n){background:linear-gradient(135deg,#8B5CF6 0%,#1e1b4b 100%);box-shadow:0 12px 32px rgba(139,92,246,.45)}
 .card:hover{transform:translateY(-8px) scale(1.03);box-shadow:0 0 42px rgba(255,46,99,.9)}
 .label{font-size:11.5px;color:#ffffff;opacity:.95;margin-top:4px;letter-spacing:1.4px;text-transform:uppercase;font-weight:800}
 .value{font-size:28px;font-weight:900;margin-top:6px;letter-spacing:-.5px;color:#ffffff}
</style>
<div class="row" id="row"></div>
<script>
 const kpis = __DATA__, row = document.getElementById('row');
 kpis.forEach((k, i) => {
   const card = document.createElement('div'); card.className = 'card';
   card.innerHTML = '<div class="label">' + k.label + '</div><div class="value">0</div>';
   row.appendChild(card);
   const el = card.querySelector('.value'), start = performance.now() + i * 130, dur = 1700;
   function tick(now) {
     const t = Math.max(0, Math.min((now - start) / dur, 1));
     const v = k.value * (1 - Math.pow(1 - t, 3));
     el.textContent = k.prefix + v.toLocaleString(undefined, {minimumFractionDigits: k.dec, maximumFractionDigits: k.dec}) + k.suffix;
     if (t < 1) requestAnimationFrame(tick);
   }
   requestAnimationFrame(tick);
 });
</script>""".replace("__DATA__", json.dumps(kpis)), height=155)

st.caption(
    "Highest Insurance Cost = the maximum individual annual insurance charge observed "
    "within the current filter selection. It is not an average; a single outlier can define it."
)

# ============================================================
# TABS
# ============================================================
t1, t2, t3, t4, t5, t6 = st.tabs([
    "Overview",
    "Linear Regression Lab",
    "Animated Story",
    "Model Performance",
    "Predict Charges",
    "Test Custom Data"
])

# ============================================================
# TAB 1: Overview
# ============================================================
with t1:
    ins = []
    if len(s_yes) and len(s_no):
        ins.append(("Smoking - The Strongest Driver",
                    f"Average for smokers is {dual_str(s_yes.mean())} vs non-smokers at "
                    f"{dual_str(s_no.mean())} - a {s_yes.mean() / s_no.mean():.1f}x increase."))
    ob, nob = df[df['bmi'] >= 30]['charges'], df[df['bmi'] < 30]['charges']
    if len(ob) and len(nob):
        ins.append(("BMI Threshold Effect",
                    f"BMI 30+ average {dual_str(ob.mean())} vs BMI under 30 at {dual_str(nob.mean())}. "
                    "The gap widens sharply among smokers."))
    ag = df.groupby('age_group', observed=True)['charges'].mean()
    if len(ag) > 1:
        ins.append(("Cost Grows With Age",
                    f"{ag.index[0]} averages {dual_str(ag.iloc[0])}, rising to "
                    f"{dual_str(ag.iloc[-1])} at {ag.index[-1]}."))
    if ins:
        for col, (tt, tx) in zip(st.columns(len(ins)), ins):
            col.markdown(glass(tt, tx), unsafe_allow_html=True)
        st.write("")

    c1, c2 = st.columns(2)
    anim = df.groupby(['region', 'age_group'], observed=True)['charges'].mean().reset_index()
    f1 = px.bar(anim, x='age_group', y='charges', animation_frame='region',
                color='charges', range_y=[0, anim['charges'].max() * 1.15],
                color_continuous_scale=th['scale'],
                title="Average Charges by Age Group (animated across regions)")
    f1.update_coloraxes(showscale=False)
    f1.update_traces(marker_line_width=0)
    f1.update_layout(yaxis_title="Average Charges (USD)")
    f1.layout.updatemenus = ()
    add_play_button(f1, 900, "Play Animation")
    c1.plotly_chart(style_fig(f1), use_container_width=True)

    share = df['smoker'].value_counts().reset_index()
    share.columns = ['smoker', 'count']
    f2 = px.pie(share, names='smoker', values='count', hole=0.62,
                color='smoker', color_discrete_map=SMOKER_MAP,
                title="Smoker Distribution")
    f2.update_traces(textinfo='percent+label', pull=[0.08] * len(share),
                     marker=dict(line=dict(color='#fff', width=3)),
                     textfont=dict(size=14, color='#fff', family='Inter'))
    c2.plotly_chart(style_fig(f2), use_container_width=True)

    c3, c4 = st.columns(2)
    f3 = px.histogram(df, x='charges', color='smoker', nbins=40, barmode='overlay',
                      opacity=.8, color_discrete_map=SMOKER_MAP,
                      title="Charges Distribution by Smoking Status")
    f3.update_traces(marker_line_width=0)
    f3.update_layout(xaxis_title="Insurance Charges (USD)")
    c3.plotly_chart(style_fig(f3), use_container_width=True)

    enc = pd.get_dummies(df[['age', 'bmi', 'children', 'sex', 'smoker', 'region']],
                         drop_first=True, dtype=int)
    corr = enc.corrwith(df['charges']).dropna().sort_values()
    f4 = px.bar(x=corr.values, y=corr.index, orientation='h',
                color=corr.values, color_continuous_scale=th['scale'],
                title="Feature Correlation with Charges",
                labels={'x': 'Correlation', 'y': ''})
    f4.update_coloraxes(showscale=False)
    f4.update_traces(marker_line_width=0)
    c4.plotly_chart(style_fig(f4), use_container_width=True)

# ============================================================
# TAB 2: Linear Regression Lab
# ============================================================
with t2:
    st.markdown("### Linear Regression Analysis")
    st.markdown(
        '<div class="formula">charges = b0 + b1*age + b2*bmi + b3*children + b4*bmi_smoker + b5*bmi_obese + ...</div>',
        unsafe_allow_html=True
    )
    st.write("")

    c1, c2, c3 = st.columns(3)

    f = px.scatter(df, x='age', y='charges', color='smoker', opacity=.7,
                   color_discrete_map=SMOKER_MAP,
                   title="Age vs Charges with Regression Line",
                   hover_data=['bmi', 'children'])
    f.update_traces(marker=dict(size=9, line=dict(width=1, color='rgba(255,255,255,.7)')))
    add_best_fit_line(f, df['age'].to_numpy(), df['charges'].to_numpy())
    f.update_layout(yaxis_title="Charges (USD)")
    c1.plotly_chart(style_fig(f, 430), use_container_width=True)

    f = px.scatter(df, x='bmi', y='charges', color='smoker', opacity=.7,
                   color_discrete_map=SMOKER_MAP,
                   title="BMI vs Charges with Regression Line",
                   hover_data=['age', 'children'])
    f.update_traces(marker=dict(size=9, line=dict(width=1, color='rgba(255,255,255,.7)')))
    f.add_vline(x=30, line_dash='dash', line_color='#FFB800', opacity=.85,
                annotation_text="BMI 30 threshold", annotation_font_color='#FFB800',
                annotation_position="top")
    add_best_fit_line(f, df['bmi'].to_numpy(), df['charges'].to_numpy())
    f.update_layout(yaxis_title="Charges (USD)")
    c2.plotly_chart(style_fig(f, 430), use_container_width=True)

    gc = df.groupby('children')['charges'].mean().reset_index()
    gc['children'] = gc['children'].astype(str)
    f = px.bar(gc, x='children', y='charges', text_auto='.0f',
               color='charges', color_continuous_scale=th['scale'],
               title="Children vs Average Charges")
    f.update_coloraxes(showscale=False)
    f.update_traces(marker_line_width=0, textfont=dict(color='#fff', size=11))
    f.update_layout(yaxis_title="Average Charges (USD)")
    c3.plotly_chart(style_fig(f, 430), use_container_width=True)

    st.markdown("##### Categorical Breakdown")

    c4, c5, c6 = st.columns(3)
    f = px.violin(df, x='sex', y='charges', color='sex', box=True, points='all',
                  color_discrete_sequence=[SKY, PINK], title="Sex vs Charges")
    f.update_traces(marker=dict(opacity=.45, size=3), line_color='#fff', meanline_visible=True)
    c4.plotly_chart(style_fig(f, 400), use_container_width=True)

    f = px.violin(df, x='smoker', y='charges', color='smoker', box=True, points='all',
                  color_discrete_map=SMOKER_MAP, title="Smoker vs Charges")
    f.update_traces(marker=dict(opacity=.45, size=3), line_color='#fff', meanline_visible=True)
    c5.plotly_chart(style_fig(f, 400), use_container_width=True)

    f = px.box(df, x='region', y='charges', color='region', points='outliers',
               color_discrete_map=REGION_MAP, title="Region vs Charges")
    c6.plotly_chart(style_fig(f, 400), use_container_width=True)

    c7, c8 = st.columns(2)
    heat = df.pivot_table(index='age_group', columns='region', values='charges',
                          aggfunc='mean', observed=True)
    heat.index = heat.index.astype(str)
    f = px.imshow(heat, text_auto='.0f', aspect='auto', color_continuous_scale=th['scale'],
                  title="Average Charges: Age Group x Region")
    c7.plotly_chart(style_fig(f), use_container_width=True)

    f = px.sunburst(df, path=['region', 'smoker', 'bmi_class'], values='charges',
                    color='charges', color_continuous_scale=th['scale'],
                    title="Region / Smoker / BMI Class Hierarchy")
    f.update_coloraxes(showscale=False)
    c8.plotly_chart(style_fig(f), use_container_width=True)

# ============================================================
# TAB 3: Animated Story
# ============================================================
with t3:
    st.markdown("### Animated Storytelling")
    c1, c2 = st.columns(2)

    d3 = df.sort_values('age_group')
    f = px.scatter(d3, x='bmi', y='charges', color='smoker', size='age',
                   animation_frame='age_group', opacity=.85,
                   color_discrete_map=SMOKER_MAP, size_max=22,
                   range_x=[df['bmi'].min() - 1, df['bmi'].max() + 1],
                   range_y=[0, df['charges'].max() * 1.15],
                   title="BMI vs Charges (bubble size = age, animated by age group)")
    f.update_traces(marker=dict(line=dict(width=1.5, color='#fff')))
    f.add_vline(x=30, line_dash='dash', line_color='#FFB800', opacity=.85,
                annotation_text="BMI 30")
    f.update_layout(yaxis_title="Charges (USD)")
    f.layout.updatemenus = ()
    add_play_button(f, 900, "Play Animation")
    c1.plotly_chart(style_fig(f, 480), use_container_width=True)

    g = df.groupby(['age', 'smoker'])['charges'].mean().reset_index()
    subs = {s: g[g['smoker'] == s].sort_values('age') for s in g['smoker'].unique()}
    n = max(len(v) for v in subs.values())

    def traces(i):
        return [go.Scatter(
            x=v['age'][:i], y=v['charges'][:i], mode='lines+markers',
            name=f'Smoker: {s}',
            line=dict(color=SMOKER_MAP.get(s, SKY), width=3.5),
            marker=dict(size=9, line=dict(width=2, color='#fff'))
        ) for s, v in subs.items()]

    f = go.Figure(data=traces(2),
                  frames=[go.Frame(data=traces(i), name=str(i)) for i in range(2, n + 1)])
    f.update_layout(
        title="Average Charges by Age (line drawn progressively)",
        xaxis=dict(range=[df['age'].min() - 1, df['age'].max() + 1], title='Age'),
        yaxis=dict(range=[0, g['charges'].max() * 1.15], title='Average Charges (USD)')
    )
    add_play_button(f, 120, "Play Animation")
    c2.plotly_chart(style_fig(f, 480), use_container_width=True)

    d3s = df.sample(min(len(df), 1500), random_state=1)
    f = go.Figure(go.Scatter3d(
        x=d3s['age'], y=d3s['bmi'], z=d3s['charges'], mode='markers',
        marker=dict(size=4.5, color=d3s['charges'], colorscale=th['scale'],
                    opacity=.9, line=dict(width=0.5, color='#fff')),
        hovertemplate='Age: %{x}<br>BMI: %{y:.1f}<br>Charges: $%{z:,.0f}<extra></extra>'
    ))
    f.frames = [go.Frame(layout=dict(scene_camera=dict(
        eye=dict(x=2 * np.cos(t), y=2 * np.sin(t), z=0.85))))
        for t in np.linspace(0, 2 * np.pi, 80)]
    f.update_layout(
        title="3D View (Age x BMI x Charges, auto-rotating)",
        scene=dict(
            xaxis_title='Age', yaxis_title='BMI', zaxis_title='Charges (USD)',
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor=th['grid'],
                       tickfont=dict(color=th['text'])),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor=th['grid'],
                       tickfont=dict(color=th['text'])),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor=th['grid'],
                       tickfont=dict(color=th['text'])),
        )
    )
    add_play_button(f, 60, "Play Animation")
    st.plotly_chart(style_fig(f, 600), use_container_width=True)

# ============================================================
# TAB 4: Model Performance
# ============================================================
with t4:
    st.markdown("### Model Performance - Linear Regression")
    st.caption("Target variable: log1p(charges). Features: age, bmi, children, one-hot categoricals + engineered bmi_smoker & bmi_obese")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(metric_card("CV R2 (5-Fold)", f"{M['cv_mean']:.3f}", f"+/- {M['cv_std']:.3f}"), unsafe_allow_html=True)
    m2.markdown(metric_card("Test R2 (Log)", f"{M['r2_log']:.3f}", f"{M['n_test']} unseen rows"), unsafe_allow_html=True)
    m3.markdown(metric_card("Test R2 (USD)", f"{M['r2_usd']:.3f}", "on raw charges"), unsafe_allow_html=True)
    m4.markdown(metric_card("RMSE", f"${M['rmse']:,.0f}", f"PKR {to_pkr(M['rmse']):,.0f}"), unsafe_allow_html=True)
    m5.markdown(metric_card("MAE", f"${M['mae']:,.0f}", f"PKR {to_pkr(M['mae']):,.0f}"), unsafe_allow_html=True)
    st.write("")

    steps = [
        ("1. Encode", "One-hot encoding with drop_first=True for sex, smoker, region."),
        ("2. Feature Engineering", "bmi_smoker interaction and bmi_obese binary flag created."),
        ("3. Log Transform", "log1p(charges) reduces skewness and stabilizes variance."),
        ("4. Validate", "80/20 train-test split + 5-fold cross-validation."),
    ]
    for col, (tt, tx) in zip(st.columns(4), steps):
        col.markdown(glass(tt, tx), unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    f = go.Figure()
    f.add_trace(go.Scatter(
        x=act_d, y=pred_d, mode='markers', name='Test Predictions',
        marker=dict(color=RED, size=9, opacity=.7,
                    line=dict(width=1, color='rgba(255,255,255,.8)')),
        hovertemplate='Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>'
    ))
    f.add_trace(go.Scatter(
        x=[act_d.min(), act_d.max()], y=[act_d.min(), act_d.max()],
        mode='lines', name='Perfect Prediction',
        line=dict(color='#FFB800', dash='dash', width=3)
    ))
    f.update_layout(title="Actual vs Predicted Charges",
                    xaxis_title="Actual (USD)", yaxis_title="Predicted (USD)")
    c1.plotly_chart(style_fig(f, 460), use_container_width=True)

    coef = pd.Series(model.coef_, index=cols).sort_values()
    f = px.bar(x=coef.values, y=coef.index, orientation='h',
               color=coef.values > 0,
               color_discrete_map={True: RED, False: SKY},
               title="Model Coefficients (effect on log charges)",
               labels={'x': 'Coefficient', 'y': ''})
    f.update_layout(showlegend=False)
    f.update_traces(marker_line_width=0)
    c2.plotly_chart(style_fig(f, 460), use_container_width=True)

    res = act_d - pred_d
    f = px.histogram(x=res, nbins=45, color_discrete_sequence=[PURPLE],
                     title="Residuals (Actual - Predicted)",
                     labels={'x': 'Error (USD)'})
    f.update_traces(marker_line_width=0)
    f.add_vline(x=0, line_dash='dash', line_color='#FFB800', line_width=2)
    st.plotly_chart(style_fig(f, 380), use_container_width=True)

# ============================================================
# TAB 5: Predict Charges
# ============================================================
with t5:
    st.markdown("### Predict Individual Insurance Charges")
    left, right = st.columns([1, 1.3])

    with left:
        st.markdown("##### Profile Inputs")
        p_age = st.slider("Age", 18, 64, 35)
        p_bmi = st.slider("BMI", 15.0, 50.0, 27.0, step=0.1)
        p_kids = st.select_slider("Children", options=[0, 1, 2, 3, 4, 5], value=0)
        k1, k2 = st.columns(2)
        p_sex = k1.selectbox("Sex", sorted(data['sex'].unique()))
        p_smoker = k2.radio("Smoker", sorted(data['smoker'].unique()), horizontal=True)
        p_region = st.selectbox("Region", sorted(data['region'].unique()))

    pred = predict_row(model, cols, p_age, p_bmi, p_kids, p_sex, p_smoker, p_region)
    p_no = predict_row(model, cols, p_age, p_bmi, p_kids, p_sex, 'no', p_region)
    p_yes = predict_row(model, cols, p_age, p_bmi, p_kids, p_sex, 'yes', p_region)

    with right:
        st.markdown(f"""
        <div class="pred-banner">
            <div class="meta">Estimated Yearly Insurance Charges</div>
            <div class="amt">${pred:,.0f}</div>
            <div class="amt-pkr">PKR {to_pkr(pred):,.0f}</div>
            <div class="meta">Dataset average: {dual_str(data['charges'].mean())}</div>
        </div>
        """, unsafe_allow_html=True)

        top = max(float(data['charges'].max()), pred * 1.1)
        g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            number={'prefix': '$', 'font': {'size': 44, 'color': RED, 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, top], 'tickcolor': th['text']},
                'bar': {'color': RED, 'thickness': 0.32},
                'steps': [
                    {'range': [0, 10000], 'color': 'rgba(0,194,255,.3)'},
                    {'range': [10000, 25000], 'color': 'rgba(255,184,0,.3)'},
                    {'range': [25000, top], 'color': 'rgba(255,46,99,.3)'}
                ],
                'threshold': {
                    'line': {'color': "#fff", 'width': 4},
                    'thickness': 0.85,
                    'value': pred
                }
            }
        ))
        st.plotly_chart(style_fig(g, 300), use_container_width=True)

    f = px.bar(
        x=['Non-Smoker', 'Smoker'], y=[p_no, p_yes],
        color=['no', 'yes'], color_discrete_map=SMOKER_MAP,
        text_auto='$,.0f',
        title="Impact of Smoking (Same Profile Comparison)",
        labels={'x': '', 'y': 'Predicted Charges (USD)'}
    )
    f.update_layout(showlegend=False)
    f.update_traces(marker_line_width=0, textfont=dict(color='#fff', size=13))
    st.plotly_chart(style_fig(f, 380), use_container_width=True)
    st.info(f"Smoking increases predicted charges by approximately "
            f"{dual_str(p_yes - p_no)} for this profile.")

# ============================================================
# TAB 6: Test Custom Data
# ============================================================
with t6:
    st.markdown("### Test Your Own Data")
    st.caption("Upload a CSV or Excel file with columns: age, sex, bmi, children, smoker, region "
               "(and optionally charges for R2 scoring). The model will predict each row.")

    up = st.file_uploader("Upload your dataset (.csv or .xlsx)",
                          type=['csv', 'xlsx', 'xls'])

    st.markdown("##### Or manually enter a single new patient row")

    with st.form("manual_test", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns(4)
        m_age = c1.number_input("Age", 18, 100, 35, step=1)
        m_sex = c2.selectbox("Sex", ['male', 'female'])
        m_bmi = c3.number_input("BMI", 10.0, 60.0, 27.0, step=0.1)
        m_kids = c4.number_input("Children", 0, 10, 0, step=1)
        c5, c6, c7 = st.columns(3)
        m_smoker = c5.selectbox("Smoker", ['no', 'yes'])
        m_region = c6.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])
        m_actual = c7.number_input("Actual Charges (USD) - optional", 0.0, 200000.0, 0.0, step=100.0)
        submit = st.form_submit_button("Predict for This Patient", use_container_width=True)

    if submit:
        p = predict_row(model, cols, m_age, m_bmi, m_kids, m_sex, m_smoker, m_region)
        st.markdown(f"""
        <div class="pred-banner">
            <h2>Prediction for the entered patient</h2>
            <div class="amt">${p:,.0f}</div>
            <div class="amt-pkr">PKR {to_pkr(p):,.0f}</div>
            <div class="meta">Profile: {m_age} yrs / {m_sex} / BMI {m_bmi} / {m_kids} children / {m_smoker} smoker / {m_region}</div>
        </div>
        """, unsafe_allow_html=True)

        if m_actual > 0:
            err = m_actual - p
            pct = (err / m_actual) * 100 if m_actual else 0
            score = max(0, 100 - abs(pct))
            st.markdown(
                f"**Actual:** {usd_str(m_actual)}  |  **Predicted:** {usd_str(p)}  |  "
                f"**Difference:** {usd_str(err)} ({pct:+.1f}%)  |  **Accuracy:** {score:.1f}%"
            )
            if abs(pct) < 10:
                st.success("Excellent prediction accuracy.")
            elif abs(pct) < 25:
                st.info("Good prediction - within acceptable range.")
            else:
                st.warning("Prediction differs significantly. Consider checking input values.")

    if up is not None:
        try:
            new_df = pd.read_csv(up) if up.name.endswith('.csv') else pd.read_excel(up)
            st.success(f"Loaded {len(new_df):,} rows with columns: {list(new_df.columns)}")

            required = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
            missing = [c for c in required if c not in new_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                with st.spinner("Running Linear Regression predictions..."):
                    preds = predict_dataframe(model, cols, new_df[required].copy())
                out = new_df.copy()
                out['predicted_charges_usd'] = preds
                out['predicted_charges_pkr'] = preds * USD_TO_PKR

                st.markdown("#### Predictions Preview")
                st.dataframe(out.head(50), use_container_width=True)

                if 'charges' in new_df.columns:
                    actual = new_df['charges'].values
                    r2 = r2_score(actual, preds)
                    rmse = float(np.sqrt(mean_squared_error(actual, preds)))
                    mae = float(mean_absolute_error(actual, preds))
                    mape = float(np.mean(np.abs((actual - preds) / np.where(actual == 0, 1, actual)))) * 100
                    acc_pct = max(0, 100 - mape)

                    st.markdown("#### Model Accuracy on Your Uploaded Data")
                    a1, a2, a3, a4, a5 = st.columns(5)
                    a1.markdown(metric_card("R2 Score", f"{r2:.4f}", "closer to 1 is better"), unsafe_allow_html=True)
                    a2.markdown(metric_card("RMSE", f"${rmse:,.0f}", f"PKR {to_pkr(rmse):,.0f}"), unsafe_allow_html=True)
                    a3.markdown(metric_card("MAE", f"${mae:,.0f}", f"PKR {to_pkr(mae):,.0f}"), unsafe_allow_html=True)
                    a4.markdown(metric_card("MAPE", f"{mape:.2f}%", "mean absolute percent error"), unsafe_allow_html=True)
                    a5.markdown(metric_card("Accuracy", f"{acc_pct:.2f}%", "100 minus MAPE"), unsafe_allow_html=True)
                    st.write("")

                    c1, c2 = st.columns(2)
                    f = go.Figure()
                    f.add_trace(go.Scatter(
                        x=actual, y=preds, mode='markers', name='Predictions',
                        marker=dict(color=RED, size=9, opacity=.7,
                                    line=dict(width=1, color='rgba(255,255,255,.8)')),
                        hovertemplate='Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>'
                    ))
                    f.add_trace(go.Scatter(
                        x=[actual.min(), actual.max()],
                        y=[actual.min(), actual.max()],
                        mode='lines', name='Perfect Prediction',
                        line=dict(color='#FFB800', dash='dash', width=3)
                    ))
                    f.update_layout(title="Actual vs Predicted on Your Data",
                                    xaxis_title="Actual (USD)",
                                    yaxis_title="Predicted (USD)")
                    c1.plotly_chart(style_fig(f, 460), use_container_width=True)

                    residuals = actual - preds
                    f2 = px.histogram(x=residuals, nbins=45,
                                      color_discrete_sequence=[PURPLE],
                                      title="Residual Distribution",
                                      labels={'x': 'Error (USD)'})
                    f2.add_vline(x=0, line_dash='dash', line_color='#FFB800', line_width=2)
                    f2.update_traces(marker_line_width=0)
                    c2.plotly_chart(style_fig(f2, 460), use_container_width=True)

                    c3, c4 = st.columns(2)
                    sample = out.sample(min(len(out), 800), random_state=1)
                    f3 = px.scatter(sample, x='age', y='predicted_charges_usd',
                                    color='smoker', opacity=.75,
                                    color_discrete_map=SMOKER_MAP,
                                    title="Predicted Charges vs Age (your data)",
                                    hover_data=['bmi', 'children'])
                    f3.update_traces(marker=dict(size=9, line=dict(width=1, color='rgba(255,255,255,.7)')))
                    add_best_fit_line(f3, sample['age'].to_numpy(),
                                      sample['predicted_charges_usd'].to_numpy())
                    f3.update_layout(yaxis_title="Predicted Charges (USD)")
                    c3.plotly_chart(style_fig(f3, 430), use_container_width=True)

                    f4 = px.scatter(sample, x='bmi', y='predicted_charges_usd',
                                    color='smoker', opacity=.75,
                                    color_discrete_map=SMOKER_MAP,
                                    title="Predicted Charges vs BMI (your data)",
                                    hover_data=['age', 'children'])
                    f4.update_traces(marker=dict(size=9, line=dict(width=1, color='rgba(255,255,255,.7)')))
                    add_best_fit_line(f4, sample['bmi'].to_numpy(),
                                      sample['predicted_charges_usd'].to_numpy())
                    f4.update_layout(yaxis_title="Predicted Charges (USD)")
                    c4.plotly_chart(style_fig(f4, 430), use_container_width=True)
                else:
                    st.info("No 'charges' column found - showing predictions only. "
                            "Add a 'charges' column to see R2 and accuracy metrics.")

                csv = out.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Predictions as CSV",
                    csv, "predictions.csv", "text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Could not process the file: {e}")

st.caption("Built with Python / Pandas / Plotly / Streamlit / scikit-learn - Linear Regression Studio")