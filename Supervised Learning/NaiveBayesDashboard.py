"""SKYCAST AI - Weather Intelligence (Gaussian Naive Bayes).
Single file. Needs train_80.csv and test_20.csv in the same folder.
Run: python -m streamlit run NaiveBayesDashboard.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.naive_bayes import GaussianNB

BASE = Path(__file__).resolve().parent
TRAIN_CSV, TEST_CSV = BASE / "train_80.csv", BASE / "test_20.csv"
ARTIFACT = BASE / "artifacts" / "skycast_model.joblib"
GITHUB_URL, LINKEDIN_URL = "", ""

NUM = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
       "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am",
       "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"]
CAT = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm"]
REQUIRED = NUM + CAT + ["RainToday"]
YESNO = {"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TIPS = {
    "Rainfall": "Rain recorded in the 24 hours to 9am (mm).", "Evaporation": "Class A pan evaporation (mm).",
    "Sunshine": "Bright sunshine hours in the day.", "Cloud9am": "Cloud cover in oktas (0 clear, 8 overcast).",
    "Cloud3pm": "Cloud cover in oktas (0 clear, 8 overcast).", "Pressure9am": "Pressure at 9am, reduced to sea level (hPa).",
    "Pressure3pm": "Pressure at 3pm, reduced to sea level (hPa).", "WindGustSpeed": "Strongest wind gust of the day (km/h).",
    "MinTemp": "Overnight minimum temperature (°C).", "MaxTemp": "Daytime maximum temperature (°C).",
    "Humidity9am": "Relative humidity at 9am (%).", "Humidity3pm": "Relative humidity at 3pm (%).",
}
NAV = {"Overview": "o", "Rain Prediction": "p", "Weather Analytics": "a", "Model Performance": "m",
       "Batch Predictions": "b", "About the Project": "i"}

try:
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
    STRETCH = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}
except Exception:
    STRETCH = {"use_container_width": True}
PLOT = dict(STRETCH, theme=None, config={"displayModeBar": False, "responsive": True})

THEMES = {
    "dark": dict(
        bg="#050E1E", sidebar="#08182C", card="#0F2440", elev="#15304F", plot="#081A2F",
        blue="#4C8DFF", cyan="#38E0F0", green="#8FE04C",
        text="#F2F6FB", muted="#A9BCD2", border="#26405C",
        btn="#04122A", shadow="0 10px 28px rgba(0,0,0,.35)", glow="rgba(76,141,255,.22)",
        rain="#123A66", dry="#12371F", scale0="#081A2F", tpl="plotly_dark",
        warm="#F5A524", grad_a="#4C8DFF", grad_b="#38E0F0",
    ),
    "light": dict(
        bg="#EEF3F9", sidebar="#FFFFFF", card="#FFFFFF", elev="#E4ECF6", plot="#FFFFFF",
        blue="#1D4ED8", cyan="#0E7490", green="#3F6212",
        text="#0B2545", muted="#42566E", border="#CBD8E6",
        btn="#FFFFFF", shadow="0 8px 22px rgba(11,37,69,.10)", glow="rgba(29,78,216,.16)",
        rain="#DCE9FB", dry="#E6F1DA", scale0="#E4ECF6", tpl="plotly_white",
        warm="#B45309", grad_a="#1D4ED8", grad_b="#0E7490",
    ),
}


def target_series(s):
    return s.astype(str).str.strip().str.lower().map(YESNO)


def fit_config(train):
    cfg = {"medians": {c: float(pd.to_numeric(train[c], errors="coerce").median()) for c in NUM},
           "mins": {c: float(pd.to_numeric(train[c], errors="coerce").min()) for c in NUM},
           "maxs": {c: float(pd.to_numeric(train[c], errors="coerce").max()) for c in NUM},
           "modes": {c: str(train[c].mode()[0]) for c in CAT},
           "categories": {c: sorted(train[c].dropna().astype(str).unique()) for c in CAT}}
    cfg["month_median"] = float(pd.to_datetime(train["Date"], errors="coerce").dt.month.median())
    cfg["rain_today_mode"] = int(target_series(train["RainToday"]).mode()[0])
    cfg["features"] = NUM + ["RainToday", "Month"] + [f"{c}_{v}" for c in CAT for v in cfg["categories"][c]]
    return cfg


def preprocess(df, cfg):
    d = df.copy()
    month = d["Month"] if "Month" in d.columns else pd.to_datetime(d["Date"], errors="coerce").dt.month
    X = pd.DataFrame(index=d.index)
    for c in NUM:
        X[c] = pd.to_numeric(d[c], errors="coerce").fillna(cfg["medians"][c])
    X["RainToday"] = target_series(d["RainToday"]).fillna(cfg["rain_today_mode"]).astype(int)
    X["Month"] = pd.to_numeric(month, errors="coerce").fillna(cfg["month_median"])
    dummies = {}
    for c in CAT:
        s = d[c].astype(str).str.strip()
        s = s.where(s.isin(cfg["categories"][c]), cfg["modes"][c])
        for v in cfg["categories"][c]:
            dummies[f"{c}_{v}"] = (s == v).astype(int)
    return pd.concat([X, pd.DataFrame(dummies, index=d.index)], axis=1)[cfg["features"]]


@st.cache_resource(show_spinner="Preparing the model (first launch only)...")
def load_artifacts():
    if ARTIFACT.exists():
        return joblib.load(ARTIFACT)
    train = pd.read_csv(TRAIN_CSV).dropna(subset=["RainTomorrow"])
    cfg = fit_config(train)
    model = GaussianNB().fit(preprocess(train, cfg), target_series(train["RainTomorrow"]).astype(int))
    ARTIFACT.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "cfg": cfg}, ARTIFACT)
    return {"model": model, "cfg": cfg}


def predict_proba_rain(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


@st.cache_resource(show_spinner=False)
def load_raw(path):
    df = pd.read_csv(path)
    df["Month"] = pd.to_datetime(df["Date"], errors="coerce").dt.month
    return df


@st.cache_data(show_spinner=False)
def evaluate_test():
    art = load_artifacts()
    test = load_raw(TEST_CSV).dropna(subset=["RainTomorrow"])
    y = target_series(test["RainTomorrow"]).astype(int)
    return y.to_numpy(), art["model"].predict(preprocess(test, art["cfg"])), len(test)


def metric_dict(y, p):
    return {"Accuracy": accuracy_score(y, p), "Precision": precision_score(y, p, zero_division=0),
            "Recall": recall_score(y, p, zero_division=0), "F1-score": f1_score(y, p, zero_division=0)}


def batch_predict(df, cfg, model):
    missing = [c for c in REQUIRED if c not in df.columns]
    if "Date" not in df.columns and "Month" not in df.columns:
        missing.append("Date or Month")
    if missing:
        raise ValueError("These required columns are missing: " + ", ".join(missing))
    proba = predict_proba_rain(model, preprocess(df, cfg))
    out = df.copy()
    out["Predicted_RainTomorrow"] = np.where(proba >= 0.5, "Yes", "No")
    out["Probability_No_Rain"] = (1 - proba).round(4)
    out["Probability_Rain"] = proba.round(4)
    return out, proba


def init_theme():
    q = st.query_params.get("theme", "dark")
    q = q[0] if isinstance(q, list) and q else q
    if "theme_choice" not in st.session_state:
        st.session_state.theme_choice = "Light Mode" if q == "light" else "Dark Mode"


def theme_key():
    return "light" if st.session_state.get("theme_choice") == "Light Mode" else "dark"


def native_theme():
    try:
        t = st.context.theme.type
        if t in ("light", "dark"):
            return t
    except Exception:
        pass
    return "dark" if st.get_option("theme.base") == "dark" else "light"


STATIC_CSS = r"""
html, body, .stApp { font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif; }
html { scroll-padding-top: 5.5rem; }
.stApp {
    background: radial-gradient(1000px 460px at 88% -10%, var(--glow), transparent), var(--bg) !important;
    color: var(--text); overflow: visible !important; min-height: 100vh;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"], section.main, .main {
    background: transparent !important; color: var(--text);
    overflow: visible !important; height: auto !important;
}
[data-testid="stHeader"], .stAppHeader, header {
    background: transparent !important; box-shadow: none !important; border: 0 !important;
}
[data-testid="stHeader"] * { color: var(--text) !important; }
[data-testid="stDecoration"], [data-testid="stAppDeployButton"], .stDeployButton, #MainMenu, footer { display: none !important; }
.block-container {
    padding: 5.4rem 1.8rem 3rem !important; max-width: 1400px !important;
    overflow: visible !important; height: auto !important;
}
.stApp :is(h1,h2,h3,h4,h5,h6,[id]) { scroll-margin-top: 5.5rem; }
.stApp :is(h1,h2,h3,h4,h5,h6,p,li,label,span,div,small,td,th,summary,input,textarea,button) { color: inherit; }
.stApp :is(h1,h2,h3,h4,p,li,label,span,div) { overflow-wrap: anywhere; }
.stApp :is(h1,h2,h3,h4) {
    font-family: Inter, "Segoe UI", system-ui, sans-serif !important;
    line-height: 1.28 !important; overflow: visible !important; font-weight: 700; letter-spacing: -.01em;
}
.stApp p, .stApp li, .stApp label, .stApp div { font-weight: 400; }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
.stApp a { color: var(--blue); text-decoration: none; }
.stApp a:hover { text-decoration: underline; }
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"], [data-testid="stMarkdownContainer"], .stElementContainer { overflow: visible !important; }
section[data-testid="stSidebar"] {
    background: var(--sidebar) !important; border-right: 1px solid var(--border); color: var(--text);
}
section[data-testid="stSidebar"] > div, [data-testid="stSidebarContent"] { background: transparent !important; }
.stApp .brand { padding: 6px 2px 16px; border-bottom: 1px solid var(--border); margin-bottom: 14px; }
.stApp .brand-kicker { color: var(--blue); font-size: .68rem; letter-spacing: .18em; font-weight: 700; text-transform: uppercase; }
.stApp .brand-name { color: var(--text); font-size: 1.35rem; font-weight: 800; margin-top: 6px; line-height: 1.25; letter-spacing: -.01em; }
.stApp .brand-sub { color: var(--muted); font-size: .82rem; margin-top: 4px; font-weight: 500; }
:is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]) {
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 7px 14px; margin: 2px 6px 2px 0; cursor: pointer;
    transition: background-color .15s, border-color .15s, color .15s;
    color: var(--text); font-weight: 500;
}
:is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]) > div:not(:has([data-testid="stMarkdownContainer"])),
label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]) { display: none; }
:is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]):hover { border-color: var(--blue); }
:is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]):has(input:checked) {
    background: var(--blue); border-color: var(--blue); color: #FFFFFF; font-weight: 600;
}
:is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]):has(input:focus-visible) { outline: 2px solid var(--cyan); outline-offset: 2px; }
section[data-testid="stSidebar"] [role="radiogroup"], section[data-testid="stSidebar"] [role="radiogroup"] > div { width: 100%; }
section[data-testid="stSidebar"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]) {
    display: flex; align-items: center; width: 100%; margin: 6px 0; min-height: 44px; padding: 10px 14px; font-size: .95rem;
}
section[data-testid="stSidebar"] :is(label[data-baseweb="radio"], label[data-testid="stRadioOption"]):has(input:checked) {
    background: var(--elev); border-color: var(--blue); color: var(--text); box-shadow: inset 4px 0 0 var(--blue); font-weight: 600;
}
.stApp .hero {
    display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 320px) auto;
    align-items: center; gap: 24px 28px; margin: 0 0 4px; position: relative; z-index: 2;
}
.stApp .hero-copy { min-width: 0; }
.stApp .brand-title {
    font-size: 2.6rem; font-weight: 800; line-height: 1.12; letter-spacing: -.025em; margin: 0 0 8px;
    color: var(--blue);
    background: linear-gradient(90deg, var(--grad_a) 0%, var(--grad_b) 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    display: inline-block; padding-bottom: 2px;
}
@supports not (-webkit-background-clip: text) {
    .stApp .brand-title { color: var(--blue); -webkit-text-fill-color: currentColor; }
}
.stApp .brand-tagline { color: var(--text); font-size: 1.02rem; font-weight: 500; margin: 0 0 12px; line-height: 1.5; }
.stApp .brand-badge {
    display: inline-flex; align-items: center; gap: 6px;
    color: var(--text); background: var(--card); border: 1px solid var(--border);
    font-size: .74rem; font-weight: 700; letter-spacing: .10em; text-transform: uppercase;
    padding: 6px 12px; border-radius: 999px; line-height: 1.2;
}
.stApp .brand-badge::before {
    content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 0 3px var(--glow);
}
.stApp .hero-art { display: flex; justify-content: center; align-items: center; }
.stApp .hero-art svg { width: 100%; max-width: 320px; height: auto; display: block; }
.stApp .hero-toggle { display: flex; flex-direction: column; align-items: flex-end; gap: 6px; min-width: 220px; }
.stApp .theme-label { color: var(--muted); font-size: .72rem; font-weight: 700; letter-spacing: .10em; text-transform: uppercase; text-align: right; }
.rule { height: 2px; margin: 18px 0 22px; border-radius: 2px; background: linear-gradient(90deg, var(--blue), var(--cyan) 40%, transparent); }
@media (max-width: 1100px) {
    .stApp .hero { grid-template-columns: minmax(0, 1fr) auto; }
    .stApp .hero-art { display: none; }
}
@media (max-width: 760px) {
    .stApp .hero { grid-template-columns: 1fr; }
    .stApp .hero-toggle { align-items: flex-start; min-width: 0; }
    .stApp .theme-label { text-align: left; }
    .stApp .brand-title { font-size: 2rem; }
}
.kpi-row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin: 6px 0 22px; }
.kpi {
    position: relative; background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px 18px 16px; min-height: 132px;
    box-shadow: var(--shadow); transition: border-color .15s, box-shadow .15s;
}
.kpi::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 4px; border-radius: 14px 14px 0 0; background: var(--blue); }
.kpi:nth-child(2)::before { background: var(--cyan); }
.kpi:nth-child(3)::before { background: var(--green); }
.kpi:nth-child(4)::before { background: linear-gradient(90deg, var(--blue), var(--cyan)); }
.kpi:hover { border-color: var(--blue); box-shadow: 0 0 0 3px var(--glow), var(--shadow); }
.stApp .kpi-top { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: .74rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; line-height: 1.4; }
.stApp .kpi-value { color: var(--blue); font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace !important; font-size: 1.7rem; font-weight: 700; margin: 10px 0 6px; line-height: 1.2; }
.stApp .kpi-hint { color: var(--muted); font-size: .82rem; line-height: 1.5; font-weight: 400; }
@media (max-width: 980px) { .kpi-row { grid-template-columns: 1fr 1fr; } }
@media (max-width: 560px) { .kpi-row { grid-template-columns: 1fr; } }
.stApp .sec-h { color: var(--text); font-size: 1.28rem; font-weight: 700; margin: 6px 0 8px; line-height: 1.3; }
.stApp .caption { color: var(--muted); font-size: .88rem; margin: 0 0 14px; line-height: 1.55; font-weight: 400; }
.stApp .sec-label { color: var(--blue); font-size: .74rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 700; margin: 16px 0 10px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
.info-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin: 12px 0 16px; }
.info-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 18px; box-shadow: var(--shadow); }
.stApp .info-card h4 { color: var(--text); margin: 0 0 8px; font-size: 1rem; font-weight: 700; }
.stApp .info-card p { color: var(--muted); margin: 0; font-size: .9rem; line-height: 1.6; font-weight: 400; }
@media (max-width: 800px) { .info-grid { grid-template-columns: 1fr; } }
.flow { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 12px 0 16px; }
.stApp .flow-step { background: var(--card); border: 1px solid var(--border); color: var(--text); border-radius: 10px; padding: 8px 14px; font-size: .85rem; font-weight: 600; }
.stApp .flow-step.accent { border-color: var(--blue); color: var(--blue); }
.stApp .flow-arrow { color: var(--cyan); font-weight: 800; }
.result { display: flex; gap: 20px; align-items: center; border-radius: 14px; padding: 20px 24px; margin: 16px 0 12px; border: 1px solid var(--border); background: var(--card); box-shadow: var(--shadow); }
.result.rain { background: var(--rain); border-color: var(--blue); }
.result.dry { background: var(--dry); border-color: var(--green); }
.stApp .result h2 { color: var(--text); margin: 0 0 6px; font-size: 1.35rem; font-weight: 700; border: 0; padding: 0; }
.stApp .result p { color: var(--muted); margin: 0; font-size: .95rem; line-height: 1.5; }
.stApp .foot { color: var(--muted); font-size: .82rem; text-align: center; margin-top: 34px; padding-top: 14px; border-top: 1px solid var(--border); line-height: 1.6; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    background: var(--blue) !important; border: 1px solid var(--blue) !important; border-radius: 10px !important;
    padding: .68rem 1.15rem !important; color: #FFFFFF !important; font-weight: 700 !important; font-size: .92rem !important;
    transition: filter .15s, box-shadow .15s;
}
.stButton > button *, .stDownloadButton > button *, [data-testid="stFormSubmitButton"] > button * { color: #FFFFFF !important; }
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover { filter: brightness(1.08); box-shadow: 0 0 0 3px var(--glow); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible, [data-testid="stFormSubmitButton"] > button:focus-visible { outline: 2px solid var(--cyan) !important; outline-offset: 2px; }
section[data-testid="stSidebar"] .stButton > button { background: var(--card) !important; border-color: var(--border) !important; }
section[data-testid="stSidebar"] .stButton > button * { color: var(--text) !important; }
:is([data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stNumberInput"], [data-testid="stTextInput"]) > div > div,
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] { background: var(--elev) !important; border-color: var(--border) !important; border-radius: 10px !important; }
:is([data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stNumberInput"]) input,
[data-baseweb="select"] *, [data-baseweb="input"] input { color: var(--text) !important; -webkit-text-fill-color: var(--text); font-weight: 500; }
[data-testid="stNumberInput"] button { background: var(--elev) !important; color: var(--text) !important; }
:is([data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stNumberInput"]):focus-within > div > div { border-color: var(--blue) !important; }
.stApp :is([data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *) { color: var(--text) !important; font-size: .88rem; font-weight: 600; }
[data-baseweb="popover"], [data-baseweb="menu"], [data-baseweb="popover"] ul, div[role="listbox"] { background: var(--card) !important; border: 1px solid var(--border) !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] li *, div[role="option"], div[role="option"] * { color: var(--text) !important; }
[data-baseweb="tag"], [data-testid="stMultiSelectTagsContainer"] span { background-color: var(--blue) !important; border-radius: 8px !important; }
[data-baseweb="tag"], [data-baseweb="tag"] *, [data-testid="stMultiSelectTagsContainer"] span,
[data-testid="stMultiSelectTagsContainer"] span * { color: #FFFFFF !important; font-weight: 600; }
[data-testid="stTooltipContent"], [data-testid="stTooltipContent"] * { background: var(--card) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }
[data-testid="stForm"] { border: 0 !important; padding: 0 !important; background: transparent !important; }
[data-testid="stExpander"] details { background: var(--card); border: 1px solid var(--border); border-radius: 12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] details[open] > summary { background: transparent !important; border-radius: 12px; }
[data-testid="stExpander"] summary * { color: var(--text) !important; font-weight: 600; }
[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: var(--muted) !important; font-weight: 600; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: var(--blue) !important; font-family: "JetBrains Mono", ui-monospace, monospace !important; font-weight: 700; }
[data-testid="stAlert"], [data-testid="stAlert"] > div { background: var(--elev) !important; border-radius: 12px; }
[data-testid="stAlert"] * { color: var(--text) !important; }
[data-testid="stFileUploaderDropzone"] { background: var(--card); border: 1px dashed var(--border); border-radius: 12px; }
[data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }
[data-testid="stFileUploaderDropzone"] button { background: var(--elev) !important; border: 1px solid var(--border) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; }
hr { border-color: var(--border) !important; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""


def inject_css(theme):
    t = THEMES[theme]
    roots = ";".join(f"--{k}:{v}" for k, v in t.items() if k != "tpl")
    fix = ('[data-testid="stDataFrame"] { filter: invert(.92) hue-rotate(180deg); }'
           if native_theme() != theme else "")
    st.markdown(
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?"
        "family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');"
        f":root,.stApp{{{roots}}}{STATIC_CSS}{fix}</style>",
        unsafe_allow_html=True,
    )


def hero_svg(t):
    cloud, sun, drop, umb, line = t["cyan"], t["warm"], t["blue"], t["blue"], t["muted"]
    bell, bar = t["blue"], t["cyan"]
    return f"""
<svg viewBox="0 0 320 200" role="img" aria-label="Weather illustration">
  <defs>
    <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{t['elev']}" stop-opacity="0.65"/>
      <stop offset="100%" stop-color="{t['elev']}" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="bellFill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{bell}" stop-opacity=".45"/>
      <stop offset="100%" stop-color="{bell}" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="320" height="200" rx="18" fill="url(#skyGrad)"/>
  <path d="M10 150 C 60 150, 70 60, 105 60 C 140 60, 150 150, 200 150"
        fill="url(#bellFill)" stroke="{bell}" stroke-width="2" opacity=".75"/>
  <line x1="10" y1="150" x2="200" y2="150" stroke="{line}" stroke-width="1" opacity=".55"/>
  <g opacity=".85">
    <rect x="212" y="118" width="14" height="32" rx="3" fill="{bar}"/>
    <rect x="232" y="102" width="14" height="48" rx="3" fill="{bar}"/>
    <rect x="252" y="130" width="14" height="20" rx="3" fill="{bar}"/>
    <rect x="272" y="112" width="14" height="38" rx="3" fill="{bar}"/>
  </g>
  <g class="wx-sun" style="transform-origin:230px 62px">
    <circle cx="230" cy="62" r="18" fill="{sun}" opacity=".9"/>
    <g stroke="{sun}" stroke-width="2.2" stroke-linecap="round" opacity=".85">
      <line x1="230" y1="30" x2="230" y2="38"/>
      <line x1="230" y1="86" x2="230" y2="94"/>
      <line x1="198" y1="62" x2="206" y2="62"/>
      <line x1="254" y1="62" x2="262" y2="62"/>
      <line x1="207" y1="39" x2="213" y2="45"/>
      <line x1="247" y1="79" x2="253" y2="85"/>
      <line x1="253" y1="39" x2="247" y2="45"/>
      <line x1="213" y1="79" x2="207" y2="85"/>
    </g>
  </g>
  <g class="wx-cloud">
    <path d="M92 118 a24 24 0 0 1 4 -46 a30 30 0 0 1 56 8 a20 20 0 0 1 -6 38 z"
          fill="{cloud}" opacity=".95"/>
  </g>
  <g stroke="{drop}" stroke-width="2.4" stroke-linecap="round">
    <line class="wx-drop" x1="106" y1="126" x2="104" y2="136"/>
    <line class="wx-drop" x1="122" y1="126" x2="120" y2="136"/>
    <line class="wx-drop" x1="138" y1="126" x2="136" y2="136"/>
    <line class="wx-drop" x1="154" y1="126" x2="152" y2="136"/>
  </g>
  <g transform="translate(48,108)">
    <path d="M0 26 a28 20 0 0 1 56 0 z" fill="{umb}" opacity=".9"/>
    <line x1="28" y1="26" x2="28" y2="48" stroke="{umb}" stroke-width="2.4" stroke-linecap="round"/>
    <path d="M28 48 a6 6 0 0 0 12 0" fill="none" stroke="{umb}" stroke-width="2.4" stroke-linecap="round"/>
  </g>
</svg>
"""


CSS_ANIM = """
<style>
@keyframes wx-spin  { to { transform: rotate(360deg); } }
@keyframes wx-float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-3px); } }
@keyframes wx-drip  { 0% { opacity: 0; transform: translateY(-3px); } 35% { opacity: 1; } 100% { opacity: 0; transform: translateY(10px); } }
.stApp .hero-art .wx-sun   { animation: wx-spin 22s linear infinite; }
.stApp .hero-art .wx-cloud { animation: wx-float 5s ease-in-out infinite; }
.stApp .hero-art .wx-drop  { animation: wx-drip 1.9s ease-in infinite; }
.stApp .hero-art .wx-drop:nth-of-type(2) { animation-delay: .35s; }
.stApp .hero-art .wx-drop:nth-of-type(3) { animation-delay: .70s; }
.stApp .hero-art .wx-drop:nth-of-type(4) { animation-delay: 1.05s; }
@media (prefers-reduced-motion: reduce) {
  .stApp .hero-art .wx-sun, .stApp .hero-art .wx-cloud, .stApp .hero-art .wx-drop { animation: none !important; }
}
</style>
"""


def icons(t):
    c, b, g = t["cyan"], t["blue"], t["green"]
    return {
        "db": f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2"><ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>',
        "test": f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{b}" stroke-width="2"><path d="M9 3h6v4H9z"/><path d="M8 7h8l1 14H7L8 7z"/><path d="M10 11h4"/></svg>',
        "acc": f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{g}" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1" fill="{g}"/></svg>',
        "rain": f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2"><path d="M7 16a5 5 0 1 1 2-9 7 7 0 0 1 12 4 4 4 0 0 1-1 7H7z"/></svg>',
    }


def result_marks(t):
    rain = (f'<svg width="64" height="64" viewBox="0 0 64 64" aria-hidden="true"><path d="M18 40a10 10 0 1 1 3-19 14 14 0 0 1 25 8 8 8 0 0 1-2 11z" '
            f'fill="{t["blue"]}"/><g stroke="{t["cyan"]}" stroke-width="3" stroke-linecap="round"><line x1="22" y1="48" x2="20" y2="56"/>'
            f'<line x1="32" y1="48" x2="30" y2="56"/><line x1="42" y1="48" x2="40" y2="56"/></g></svg>')
    rays = "".join(f'<line x1="32" y1="6" x2="32" y2="14" transform="rotate({a} 32 32)"/>' for a in range(0, 360, 45))
    dry = (f'<svg width="64" height="64" viewBox="0 0 64 64" aria-hidden="true"><circle cx="32" cy="32" r="13" fill="{t["green"]}"/>'
           f'<g stroke="{t["green"]}" stroke-width="3" stroke-linecap="round">{rays}</g></svg>')
    return rain, dry


def style_fig(fig, theme, h=390):
    t = THEMES[theme]
    ax = dict(gridcolor=t["border"], zeroline=False, linecolor=t["border"],
              tickfont=dict(color=t["muted"], size=12), title_font=dict(color=t["text"], size=13))
    fig.update_layout(
        template=t["tpl"], paper_bgcolor=t["plot"], plot_bgcolor=t["plot"],
        height=h, margin=dict(l=52, r=24, t=62, b=48),
        font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif", color=t["text"], size=12.5),
        title=dict(font=dict(color=t["text"], size=15.5), x=0.01, xanchor="left"),
        legend=dict(bgcolor=t["card"], bordercolor=t["border"], borderwidth=1, font=dict(color=t["text"], size=12)),
        colorway=[t["cyan"], t["blue"], t["green"], t["warm"]],
    )
    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)
    fig.update_coloraxes(colorbar=dict(outlinecolor=t["border"], tickfont=dict(color=t["text"])))
    fig.update_traces(hoverlabel=dict(bgcolor=t["card"], font_color=t["text"], bordercolor=t["border"]))
    return fig


def show_fig(fig, theme, h=390):
    st.plotly_chart(style_fig(fig, theme, h), **PLOT)


def note(text):
    st.markdown(f'<p class="caption">{text}</p>', unsafe_allow_html=True)


def heading(title, sub=None):
    st.markdown(f'<div class="sec-h">{title}</div>', unsafe_allow_html=True)
    if sub:
        note(sub)


def kpi_row(items, ic):
    cells = "".join(
        f'<div class="kpi"><div class="kpi-top">{ic[i]}<span>{lab}</span></div>'
        f'<div class="kpi-value">{val}</div><div class="kpi-hint">{hint}</div></div>'
        for lab, val, hint, i in items
    )
    st.markdown(f'<div class="kpi-row">{cells}</div>', unsafe_allow_html=True)


def footer():
    links = " | ".join(f"<a href='{u}'>{n}</a>" for n, u in (("GitHub", GITHUB_URL), ("LinkedIn", LINKEDIN_URL)) if u)
    st.markdown(f'<div class="foot">SKYCAST AI is an educational Machine Learning project, not an official weather service. {links}</div>',
                unsafe_allow_html=True)


def _has_va():
    try:
        st.columns([1, 1], vertical_alignment="center")
        return True
    except TypeError:
        return False


def render_chrome():
    t = THEMES[theme_key()]
    st.markdown(CSS_ANIM, unsafe_allow_html=True)
    if _has_va():
        left, right = st.columns([4.0, 1.2], vertical_alignment="center")
    else:
        left, right = st.columns([4.0, 1.2])
    with left:
        st.markdown(
            f'<div class="hero">'
            f'  <div class="hero-copy">'
            f'    <h1 class="brand-title">SKYCAST AI</h1>'
            f'    <p class="brand-tagline">Predict Tomorrow. Prepare Today.</p>'
            f'    <span class="brand-badge">Gaussian Naive Bayes</span>'
            f'  </div>'
            f'  <div class="hero-art">{hero_svg(t)}</div>'
            f'  <div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="theme-label" style="text-align:right">Appearance</div>', unsafe_allow_html=True)
        st.radio("Appearance", ["Dark Mode", "Light Mode"], key="theme_choice",
                 horizontal=True, label_visibility="collapsed")
    if st.query_params.get("theme") != theme_key():
        st.query_params["theme"] = theme_key()
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)


def page_overview(theme, train, n_test, y, p):
    t, ic = THEMES[theme], icons(THEMES[theme])
    kpi_row([("Training Dataset Size", f"{len(train):,}", "Rows used to fit Gaussian Naive Bayes", "db"),
             ("Test Dataset Size", f"{n_test:,}", "Held-out labelled rows for evaluation", "test"),
             ("Model Accuracy", f"{accuracy_score(y, p):.1%}", "Correct labels on the test set", "acc"),
             ("Predicted Rain Events", f"{int(p.sum()):,}", "Test rows the model labels as rain", "rain")], ic)
    tr = train["RainTomorrow"].dropna().value_counts()
    c1, c2 = st.columns(2)
    for col, title, vals in ((c1, "Training data: what actually happened", [tr.get("No", 0), tr.get("Yes", 0)]),
                             (c2, "Test data: what the model predicts", [int((p == 0).sum()), int((p == 1).sum())])):
        fig = go.Figure(go.Pie(labels=["No rain", "Rain"], values=vals, hole=.62, sort=False, textinfo="percent",
                               marker=dict(colors=[t["cyan"], t["blue"]], line=dict(color=t["plot"], width=2))))
        fig.update_layout(title=title, legend=dict(orientation="h", y=-0.08))
        with col:
            show_fig(fig, theme, 340)
    note("Rain is the minority class, so the model trades missed rain days against false alarms. See Model Performance for details.")
    st.markdown("""<div class="info-grid">
      <div class="info-card"><h4>What it predicts</h4><p>A binary label for <b>RainTomorrow</b>: whether rain is expected the day after the observed weather.</p></div>
      <div class="info-card"><h4>What data it uses</h4><p>Temperature, rainfall, humidity, pressure, wind, cloud cover, location and month from your train and test CSV files.</p></div>
      <div class="info-card"><h4>How to use it</h4><p>Rain Prediction scores one day, Weather Analytics explores the data, Model Performance shows test metrics, Batch Predictions scores a CSV.</p></div>
    </div>""", unsafe_allow_html=True)
    note("Every number on this page is computed from your own train and test files and the trained Gaussian Naive Bayes model.")


def page_predict(theme, art):
    t, (cfg, model) = THEMES[theme], (art["cfg"], art["model"])
    rain_mark, dry_mark = result_marks(t)
    heading("Rain prediction", "Enter today's observations. Numeric defaults are training-set medians. This is a model score, not an official forecast.")
    groups = [("Temperature", ["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"]),
              ("Rainfall", ["Rainfall", "Evaporation", "Sunshine"]),
              ("Humidity", ["Humidity9am", "Humidity3pm"]),
              ("Atmospheric Pressure", ["Pressure9am", "Pressure3pm"]),
              ("Wind Conditions", ["WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"]),
              ("Cloud Conditions", ["Cloud9am", "Cloud3pm"])]
    vals = {}
    with st.form("predict_form"):
        st.markdown('<div class="sec-label">Observation context</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        locs = cfg["categories"]["Location"]
        vals["Location"] = c1.selectbox("Location", locs, index=locs.index(cfg["modes"]["Location"]), key="in_Location")
        vals["Month"] = MONTHS.index(c2.selectbox("Month", MONTHS, index=int(cfg["month_median"]) - 1, key="in_Month")) + 1
        vals["RainToday"] = c3.radio("Rain today", ["No", "Yes"], horizontal=True, key="in_RainToday")
        left, right = st.columns(2)
        for i, (title, cols) in enumerate(groups):
            with (left if i % 2 == 0 else right):
                st.markdown(f'<div class="sec-label">{title}</div>', unsafe_allow_html=True)
                for col, name in zip(st.columns(len(cols)), cols):
                    lo, hi, med = cfg["mins"][name], cfg["maxs"][name], cfg["medians"][name]
                    vals[name] = col.number_input(name, min_value=float(lo), max_value=float(hi), value=float(med),
                                                  help=TIPS.get(name, f"Observed training range {lo:.1f} to {hi:.1f}."),
                                                  key=f"in_{name}")
        st.markdown('<div class="sec-label">Wind direction</div>', unsafe_allow_html=True)
        for col, name in zip(st.columns(3), ["WindGustDir", "WindDir9am", "WindDir3pm"]):
            cats = cfg["categories"][name]
            vals[name] = col.selectbox(name, cats, index=cats.index(cfg["modes"][name]), key=f"in_{name}")
        go_btn = st.form_submit_button("PREDICT TOMORROW'S WEATHER")

    def reset():
        for k in [k for k in st.session_state if k.startswith("in_")]:
            del st.session_state[k]
        st.session_state.pop("last", None)

    st.button("Reset inputs", on_click=reset)
    if go_btn:
        with st.spinner("Running Gaussian Naive Bayes..."):
            st.session_state["last"] = (dict(vals),
                                        float(predict_proba_rain(model, preprocess(pd.DataFrame([vals]), cfg))[0]))
    if "last" in st.session_state:
        vals, p_rain = st.session_state["last"]
        rain = p_rain >= 0.5
        label = "RAIN EXPECTED TOMORROW" if rain else "NO RAIN EXPECTED TOMORROW"
        why = ("The classifier assigned a higher probability to the rain class for these observations." if rain else
               "The classifier assigned a higher probability to the no-rain class for these observations.")
        st.markdown(f'<div class="result {"rain" if rain else "dry"}">{rain_mark if rain else dry_mark}'
                    f'<div><h2>{label}</h2><p>{why}</p></div></div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Model probability: rain", f"{p_rain:.1%}")
        m2.metric("Model probability: no rain", f"{1 - p_rain:.1%}")
        fig = go.Figure(go.Bar(x=[1 - p_rain, p_rain], y=["No rain", "Rain"], orientation="h",
                               marker_color=[t["cyan"], t["blue"]],
                               text=[f"{1 - p_rain:.1%}", f"{p_rain:.1%}"], textposition="auto",
                               textfont=dict(color=t["text"])))
        fig.update_layout(title="Model class probabilities", xaxis=dict(range=[0, 1], tickformat=".0%"))
        show_fig(fig, theme, 240)
        st.warning("Educational Machine Learning prediction only. These probabilities are the model's own estimates and are "
                   "not calibrated real-world chances of rain, nor an official forecast.")
        report = ("SKYCAST AI - Prediction report\n" + "\n".join(f"{k}: {v}" for k, v in vals.items()) +
                  f"\n\nResult: {label}\nProbability rain: {p_rain:.4f}\nProbability no rain: {1 - p_rain:.4f}\n"
                  "Model: Gaussian Naive Bayes (educational project; not an official forecast)\n")
        st.download_button("Download prediction report", report, "skycast_report.txt")


@st.cache_data(show_spinner="Building charts...", max_entries=24)
def build_analytics(months, locs, cls, theme):
    t = THEMES[theme]
    df = load_raw(TRAIN_CSV).dropna(subset=["RainTomorrow"])
    if months: df = df[df["Month"].isin(months)]
    if locs: df = df[df["Location"].isin(locs)]
    df = df[df["RainTomorrow"].isin(cls)]
    if df.empty:
        return 0, {}, ""
    cmap = {"No": t["cyan"], "Yes": t["blue"]}
    f = {}
    cnt = df["RainTomorrow"].value_counts().rename_axis("RainTomorrow").reset_index(name="Count")
    cnt["Share"] = (cnt["Count"] / cnt["Count"].sum() * 100).round(1)
    fig = go.Figure(go.Pie(labels=cnt["RainTomorrow"].map({"Yes": "Rain", "No": "No rain"}), values=cnt["Count"], hole=.58,
                           marker=dict(colors=[cmap[v] for v in cnt["RainTomorrow"]], line=dict(color=t["plot"], width=2)),
                           textinfo="label+percent", textfont=dict(color=t["text"]),
                           hovertemplate="%{label}<br>%{value:,} days (%{percent})<extra></extra>"))
    fig.update_layout(title="Rain distribution: rainy vs non-rainy next days", showlegend=False)
    f["A"] = style_fig(fig, theme, 340)
    shares = ", ".join(f"{r.RainTomorrow}: {int(r.Count):,} ({r.Share}%)" for r in cnt.itertuples(index=False))
    samp = df.sample(min(len(df), 15000), random_state=7)
    f["B"] = style_fig(px.violin(samp.dropna(subset=["Humidity3pm"]), x="RainTomorrow", y="Humidity3pm",
                                 color="RainTomorrow", box=True, points=False, color_discrete_map=cmap,
                                 title="Humidity at 3pm vs next-day rain"), theme)
    r99 = samp["Rainfall"].quantile(.99)
    f["C"] = style_fig(px.box(samp[samp["Rainfall"] <= max(r99, 1)], x="RainTomorrow", y="Rainfall",
                              color="RainTomorrow", color_discrete_map=cmap,
                              title="Same-day rainfall vs next-day rain (top 1% hidden)"), theme)
    tmp = samp.melt(id_vars="RainTomorrow", value_vars=["MinTemp", "MaxTemp"],
                    var_name="Measure", value_name="Temp").dropna(subset=["Temp"])
    f["D"] = style_fig(px.violin(tmp, x="Measure", y="Temp", color="RainTomorrow", box=True,
                                 color_discrete_map=cmap,
                                 title="Minimum and maximum temperature by next-day rain class"), theme)
    m = df.assign(r=df["RainTomorrow"] == "Yes").groupby("Month")["r"].agg(["sum", "count"]).reset_index()
    m["Rain rate %"] = (m["sum"] / m["count"] * 100).round(1)
    m["Month name"] = m["Month"].apply(lambda x: MONTHS[int(x) - 1])
    fig = px.bar(m, x="Month name", y="Rain rate %", hover_data={"sum": True, "count": True},
                 title="Monthly rain patterns: share of days followed by rain")
    fig.update_traces(marker_color=t["blue"])
    f["E"] = style_fig(fig, theme, 360)
    w = df.dropna(subset=["WindGustDir"]).groupby(["WindGustDir", "RainTomorrow"]).size().reset_index(name="Count")
    f["F"] = style_fig(px.bar(w, x="WindGustDir", y="Count", color="RainTomorrow", barmode="group",
                              color_discrete_map=cmap, title="Wind gust direction and next-day rain counts"), theme, 360)
    num = df[NUM].copy()
    num["RainTomorrow (1=Yes)"] = (df["RainTomorrow"] == "Yes").astype(int)
    fig = px.imshow(num.corr().round(2), text_auto=True, aspect="auto", zmin=-1, zmax=1,
                    color_continuous_scale=[t["warm"], t["scale0"], t["cyan"], t["green"]],
                    title="Feature correlation heatmap (numeric columns)")
    fig.update_xaxes(tickangle=-45)
    f["G"] = style_fig(fig, theme, 680)
    return len(df), f, shares


def page_analytics(theme):
    heading("Weather analytics", "Charts use the training observations. Filters change what is plotted; they do not retrain the model.")
    base = load_raw(TRAIN_CSV)
    f1, f2, f3 = st.columns(3)
    months = f1.multiselect("Month", list(range(1, 13)), format_func=lambda m: MONTHS[m - 1])
    locs = f2.multiselect("Location", sorted(base["Location"].dropna().astype(str).unique()))
    cls = f3.multiselect("Rain tomorrow", ["Yes", "No"], default=["Yes", "No"])
    n, f, shares = build_analytics(tuple(months), tuple(locs), tuple(cls), theme)
    if not f:
        st.info("No rows match these filters. Clear a filter to see charts.")
        return
    st.caption(f"{n:,} training records shown")
    st.plotly_chart(f["A"], **PLOT)
    note(f"Class counts in this filter: {shares}. Because of this imbalance, accuracy alone can look strong while rain is still hard to catch.")
    c1, c2 = st.columns(2)
    c1.plotly_chart(f["B"], **PLOT)
    c1.markdown('<p class="caption">Higher humidity tends to go with rain the next day, but the overlap shows it is not decisive on its own.</p>', unsafe_allow_html=True)
    c2.plotly_chart(f["C"], **PLOT)
    c2.markdown('<p class="caption">Rain today is a useful clue for rain tomorrow, yet many dry days still follow light rain.</p>', unsafe_allow_html=True)
    st.plotly_chart(f["D"], **PLOT)
    note("Temperature distributions overlap a great deal, so heat or cold by itself is a weak separator for this task.")
    st.plotly_chart(f["E"], **PLOT)
    note("Seasonality appears as a changing rain rate by month. The model reads month as a number, not as a calendar.")
    st.plotly_chart(f["F"], **PLOT)
    note("Direction counts reflect both climate and how often each direction is recorded. Treat them as associations, not causes.")
    st.plotly_chart(f["G"], **PLOT)
    note("Gaussian Naive Bayes assumes features are independent given the class. Strong correlations (humidity, cloud, pressure) show why that is only approximate.")


def page_performance(theme):
    t, ic = THEMES[theme], icons(THEMES[theme])
    heading("Model performance", "All figures are computed on the labelled test file with the same preprocessing used at training time.")
    y, p, n = evaluate_test()
    m = metric_dict(y, p)
    kpi_row([("Accuracy", f"{m['Accuracy']:.3f}", "Share of all test predictions that match the label", "acc"),
             ("Precision", f"{m['Precision']:.3f}", "When the model predicts rain, how often it is rain", "test"),
             ("Recall", f"{m['Recall']:.3f}", "Of true rain days, how many the model caught", "rain"),
             ("F1-score", f"{m['F1-score']:.3f}", "Balance of rain-class precision and recall", "db")], ic)
    st.caption(f"Computed on {n:,} labelled test rows. Precision, recall and F1 describe the rain class.")
    st.markdown('<div class="flow"><div class="flow-step">Weather Data</div><div class="flow-arrow">→</div>'
                '<div class="flow-step">Preprocessing</div><div class="flow-arrow">→</div>'
                '<div class="flow-step accent">Gaussian Naive Bayes</div><div class="flow-arrow">→</div>'
                '<div class="flow-step">Rain Prediction</div></div>', unsafe_allow_html=True)
    note("Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem. It assumes conditional independence "
         "between features given the class and models numeric features with Gaussian distributions. It labels rain when "
         "the rain probability is at least 0.5.")
    cm = confusion_matrix(y, p, labels=[0, 1])
    c1, c2 = st.columns(2)
    with c1:
        show_fig(px.imshow(cm, text_auto=",", x=["Pred: No rain", "Pred: Rain"],
                           y=["Actual: No rain", "Actual: Rain"],
                           color_continuous_scale=[t["scale0"], t["blue"], t["cyan"]],
                           title="Confusion matrix (test set)"), theme, 400)
        note("Rows are the true label and columns the model label. Off-diagonal cells are misses and false alarms.")
    comp = pd.DataFrame({"Label": ["No rain", "Rain"] * 2,
                         "Source": ["Actual"] * 2 + ["Predicted"] * 2,
                         "Count": [int((y == 0).sum()), int((y == 1).sum()),
                                   int((p == 0).sum()), int((p == 1).sum())]})
    with c2:
        show_fig(px.bar(comp, x="Label", y="Count", color="Source", barmode="group",
                        title="Actual vs predicted labels",
                        color_discrete_sequence=[t["cyan"], t["blue"]]), theme, 400)
        note("Compare how often rain really occurs in the test file with how often the model predicts it.")
    st.markdown('<div class="sec-h">Classification report</div>', unsafe_allow_html=True)
    rep = pd.DataFrame(classification_report(y, p, target_names=["No rain", "Rain"],
                                             output_dict=True, zero_division=0)).T
    st.dataframe(rep.round(3), **STRETCH)
    st.markdown("- **Accuracy:** share of all predictions that were right.\n"
                "- **Precision:** when the model says rain, how often it is right.\n"
                "- **Recall:** of the days that really had rain, how many the model caught.\n"
                "- **F1-score:** one balanced number combining precision and recall.")
    if m["Precision"] < 0.5:
        st.info("Rain precision is below 50%, so the model raises many false alarms. Treat it as a learning experiment "
                "rather than a dependable forecaster.")


def page_batch(theme, art):
    cfg, model, ic = art["cfg"], art["model"], icons(THEMES[theme])
    heading("Batch predictions", "Upload a CSV of weather rows. The same preprocessing and Gaussian Naive Bayes model score every record.")
    st.write("Required columns: " + ", ".join(REQUIRED) + ", and either **Date** or **Month**. `RainTomorrow` is optional and is only used for evaluation, never as an input.")
    up = st.file_uploader("Weather CSV", type="csv")
    if up is None:
        st.info("Choose a CSV file to start.")
        return
    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error(f"This file could not be read as a CSV: {e}")
        return
    st.markdown('<div class="sec-h">Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), **STRETCH)
    try:
        with st.spinner("Predicting..."):
            out, proba = batch_predict(df, cfg, model)
    except ValueError as e:
        st.error(f"{e}. Add them and upload again.")
        return
    except Exception as e:
        st.error(f"The file has values the model cannot use ({type(e).__name__}: {e}). Check the column formats and try again.")
        return
    for c in CAT:
        u = sorted(set(df[c].dropna().astype(str)) - set(cfg["categories"][c]))
        if u:
            st.warning(f"{c} has values the model has never seen ({', '.join(u[:5])}). They were replaced by the training mode.")
    rainy = int((proba >= 0.5).sum())
    kpi_row([("Rows predicted", f"{len(out):,}", "Records scored in this upload", "db"),
             ("Rain expected", f"{rainy:,}", "Rows with rain probability of 0.5 or more", "rain"),
             ("No rain expected", f"{len(out) - rainy:,}", "Rows with rain probability below 0.5", "acc"),
             ("Mean P(rain)", f"{float(np.mean(proba)):.1%}", "Average rain-class probability", "test")], ic)
    st.success("Predictions are ready. This is an educational model output, not an official forecast.")
    st.dataframe(out.head(200), **STRETCH)
    st.download_button("Download results CSV", out.to_csv(index=False).encode(), "skycast_predictions.csv", "text/csv")
    if "RainTomorrow" in df.columns:
        y = target_series(df["RainTomorrow"])
        mask = y.notna()
        if mask.sum() == 0:
            st.warning("RainTomorrow has no usable Yes/No labels, so evaluation was skipped.")
        else:
            st.markdown('<div class="sec-h">Evaluation on your file</div>', unsafe_allow_html=True)
            yt, yp = y[mask].astype(int), (proba[mask.to_numpy()] >= 0.5).astype(int)
            for col, (k, v) in zip(st.columns(4), metric_dict(yt, yp).items()):
                col.metric(k, f"{v:.3f}")
            st.caption(f"{int(mask.sum()):,} rows with a known label were used; blank labels were excluded.")


def page_about():
    heading("About the project", "A compact reference for what SKYCAST AI is and how to read its outputs.")
    st.markdown("""<div class="info-grid">
      <div class="info-card"><h4>Project</h4><p><b>SKYCAST AI</b>, a weather intelligence dashboard.<br>Algorithm: Gaussian Naive Bayes<br>Task: binary classification of RainTomorrow</p></div>
      <div class="info-card"><h4>Data and stack</h4><p>Historical observations in train_80.csv and test_20.csv.<br>Framework: Streamlit. Charts: Plotly. Model: scikit-learn GaussianNB.</p></div>
      <div class="info-card"><h4>How to navigate</h4><p>Overview for context, Analytics for data patterns, Performance for test metrics, Rain Prediction for one score, Batch for CSV uploads.</p></div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="sec-h">Why Gaussian Naive Bayes</div>', unsafe_allow_html=True)
    st.write("Naive Bayes is fast, simple and outputs class probabilities, which makes it a good first probabilistic "
             "classifier. Its key assumption is that features are independent given the class. Weather variables are "
             "strongly related (humidity, cloud and pressure move together), so the assumption is only approximately "
             "true and limits accuracy.")
    st.markdown('<div class="sec-h">Workflow</div>', unsafe_allow_html=True)
    st.markdown('<div class="flow"><div class="flow-step">Weather Data</div><div class="flow-arrow">→</div>'
                '<div class="flow-step">Preprocessing</div><div class="flow-arrow">→</div>'
                '<div class="flow-step accent">Gaussian Naive Bayes</div><div class="flow-arrow">→</div>'
                '<div class="flow-step">Rain Prediction</div></div>', unsafe_allow_html=True)
    st.caption("Raw CSV, then training medians and modes for missing values, Date to Month, Yes/No to 1/0, "
               "one-hot encoding, Gaussian Naive Bayes, probabilities, label.")
    st.markdown('<div class="sec-h">Limitations</div>', unsafe_allow_html=True)
    st.markdown("- Next-day rain depends on atmospheric dynamics that one day's observations cannot fully capture.\n"
                "- Rainy days are the minority class, so overall accuracy can overstate skill on rain.\n"
                "- Medians fill gaps in Sunshine, Evaporation and Cloud columns, which can blur their signal.\n"
                "- Results hold for the stations and years in this dataset only.\n"
                "- Outputs are educational, not an official or guaranteed forecast.")
    if GITHUB_URL or LINKEDIN_URL:
        st.markdown(" | ".join(f"[{n}]({u})" for n, u in (("GitHub", GITHUB_URL), ("LinkedIn", LINKEDIN_URL)) if u))
    else:
        st.caption("Set GITHUB_URL and LINKEDIN_URL at the top of NaiveBayesDashboard.py to show your links here.")


def main():
    st.set_page_config(page_title="SKYCAST AI", page_icon="☁️", layout="wide", initial_sidebar_state="expanded")
    init_theme()
    theme = theme_key()
    inject_css(theme)
    for f in (TRAIN_CSV, TEST_CSV):
        if not f.exists():
            st.error(f"File not found: {f.name}. Place train_80.csv and test_20.csv in the same folder as NaiveBayesDashboard.py.")
            st.stop()
    art = load_artifacts()
    train = load_raw(TRAIN_CSV)
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-kicker">Weather intelligence</div>'
                    '<div class="brand-name">SKYCAST AI</div>'
                    '<div class="brand-sub">Gaussian Naive Bayes</div></div>', unsafe_allow_html=True)
        choice = NAV[st.radio("Navigate", list(NAV), label_visibility="collapsed", key="nav")]
        st.caption("Educational ML project. Not an official forecast.")
        if st.button("Retrain model", **STRETCH):
            ARTIFACT.unlink(missing_ok=True)
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
    render_chrome()
    if choice == "o":
        y, p, n = evaluate_test()
        page_overview(theme, train, n, y, p)
    elif choice == "p": page_predict(theme, art)
    elif choice == "a": page_analytics(theme)
    elif choice == "m": page_performance(theme)
    elif choice == "b": page_batch(theme, art)
    else: page_about()
    footer()


if __name__ == "__main__":
    main()

    