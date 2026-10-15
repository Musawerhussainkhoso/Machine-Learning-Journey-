"""SKYCAST AI - Weather Intelligence (Gaussian Naive Bayes). Run: streamlit run dashboard.py"""
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
GITHUB_URL, LINKEDIN_URL = "", ""  # optional: add your own links

# ---- feature schema (same final feature set as the training script: nothing dropped) ----
NUM = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
       "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am",
       "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"]
CAT = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm"]
REQUIRED = NUM + CAT + ["RainToday"]  # plus Date or Month
YESNO = {"yes": 1, "no": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
C = dict(navy="#071A2F", mid="#0B2545", blue="#2589FF", cyan="#35D6ED", green="#7CFC00", white="#F4F8FF")
TIPS = {"Rainfall": "Rain recorded in the 24 hours to 9am (mm).", "Evaporation": "Class A pan evaporation (mm).",
        "Sunshine": "Bright sunshine hours in the day.", "Cloud9am": "Cloud cover in oktas (0 clear, 8 overcast).",
        "Cloud3pm": "Cloud cover in oktas (0 clear, 8 overcast).", "Pressure9am": "Sea-level pressure (hPa).",
        "Pressure3pm": "Sea-level pressure (hPa).", "WindGustSpeed": "Strongest wind gust of the day (km/h)."}


# ---------------------------- preprocessing + training ----------------------------
def target_series(s):
    return s.astype(str).str.strip().str.lower().map(YESNO)


def fit_config(train):
    """Learn every preprocessing value from TRAINING data only."""
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
    """Apply the exact training transformations. Returns X with the model's exact columns/order."""
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
        s = s.where(s.isin(cfg["categories"][c]), cfg["modes"][c])  # unseen/missing -> training mode
        for v in cfg["categories"][c]:
            dummies[f"{c}_{v}"] = (s == v).astype(int)
    X = pd.concat([X, pd.DataFrame(dummies, index=d.index)], axis=1)
    return X[cfg["features"]]


@st.cache_resource(show_spinner="Preparing the model...")
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
    return model.predict_proba(X)[:, list(model.classes_).index(1)]  # column of class 1 = rain


@st.cache_data(show_spinner=False)
def load_raw(path):
    df = pd.read_csv(path)
    df["Month"] = pd.to_datetime(df["Date"], errors="coerce").dt.month
    return df


@st.cache_data(show_spinner=False)
def evaluate_test():
    art = load_artifacts()
    test = load_raw(TEST_CSV).dropna(subset=["RainTomorrow"])
    y = target_series(test["RainTomorrow"]).astype(int)
    X = preprocess(test, art["cfg"])
    pred = art["model"].predict(X)
    return y.to_numpy(), pred, len(X)


def metric_dict(y, p):
    return {"Accuracy": accuracy_score(y, p), "Precision": precision_score(y, p, zero_division=0),
            "Recall": recall_score(y, p, zero_division=0), "F1-score": f1_score(y, p, zero_division=0)}


# ---------------------------- styling ----------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&family=Inter:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3, .hero-title {{ font-family: 'Sora', sans-serif !important; letter-spacing: -0.01em; }}
.stApp {{ background: radial-gradient(1200px 600px at 80% -10%, #12407a55, transparent),
          linear-gradient(180deg, {C['navy']}, {C['mid']}); color: {C['white']}; }}
section[data-testid="stSidebar"] {{ background: {C['navy']}; border-right: 1px solid #35D6ED33; }}
section[data-testid="stSidebar"] label[data-baseweb="radio"] {{ padding: 6px 10px; border-radius: 10px; }}
section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {{
    background: {C['blue']}33; border-left: 3px solid {C['green']}; }}
div[data-testid="stMetric"] {{ background: #0B254588; border: 1px solid #35D6ED44; border-radius: 14px;
    padding: 14px 16px; backdrop-filter: blur(6px); transition: border-color .2s; }}
div[data-testid="stMetric"]:hover {{ border-color: {C['green']}; }}
div[data-testid="stMetricValue"] {{ color: {C['cyan']}; font-family: 'Sora', sans-serif; }}
.stButton > button, .stDownloadButton > button {{ background: linear-gradient(90deg, {C['blue']}, {C['cyan']});
    color: {C['navy']}; font-weight: 700; border: 0; border-radius: 12px; padding: .6rem 1.2rem; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ box-shadow: 0 0 0 2px {C['green']}; color: {C['navy']}; }}
.hero {{ position: relative; overflow: hidden; border: 1px solid #35D6ED55; border-radius: 22px; padding: 34px 36px;
    background: linear-gradient(120deg, #0B2545ee, #071A2Fee); margin-bottom: 18px; }}
.hero-title {{ font-size: 3.2rem; font-weight: 800; margin: 6px 0 0; color: {C['white']}; }}
.hero-sub {{ font-size: 1.25rem; color: {C['cyan']}; margin: 2px 0 10px; }}
.hero p {{ max-width: 520px; opacity: .85; }}
.badge {{ display: inline-block; font-size: .78rem; padding: 4px 12px; border-radius: 20px;
    border: 1px solid {C['green']}; color: {C['green']}; }}
.hero-art {{ position: absolute; right: 24px; top: 6px; width: 260px; height: 230px; }}
.umbrella {{ transform-origin: 130px 150px; animation: sway 4s ease-in-out infinite; }}
.drop {{ fill: {C['cyan']}; animation: fall 1.6s linear infinite; opacity: 0; }}
.cloud {{ animation: drift 7s ease-in-out infinite alternate; }}
@keyframes sway {{ 0%,100% {{ transform: rotate(-5deg); }} 50% {{ transform: rotate(5deg); }} }}
@keyframes fall {{ 0% {{ transform: translateY(-20px); opacity: 0; }} 20% {{ opacity: .9; }}
                  100% {{ transform: translateY(80px); opacity: 0; }} }}
@keyframes drift {{ from {{ transform: translateX(-10px); }} to {{ transform: translateX(14px); }} }}
@media (prefers-reduced-motion: reduce) {{ .umbrella, .drop, .cloud, .sunray {{ animation: none; }} }}
@media (max-width: 800px) {{ .hero-art {{ display: none; }} .hero-title {{ font-size: 2.2rem; }} }}
.result {{ border-radius: 18px; padding: 22px 26px; margin: 12px 0; border: 1px solid; display: flex; gap: 20px; align-items: center; }}
.result.rain {{ background: #0B2545; border-color: {C['blue']}; }}
.result.sun {{ background: linear-gradient(120deg, #0B2545, #123d2a); border-color: {C['green']}; }}
.result h2 {{ margin: 0; font-size: 1.7rem; }}
.sunray {{ transform-origin: 50px 50px; animation: spin 14s linear infinite; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.foot {{ opacity: .6; font-size: .85rem; text-align: center; margin-top: 40px; padding-top: 14px; border-top: 1px solid #35D6ED22; }}
</style>
"""

HERO = f"""
<div class="hero">
 <svg class="hero-art" viewBox="0 0 260 230" xmlns="http://www.w3.org/2000/svg">
  <g class="cloud"><ellipse cx="90" cy="48" rx="46" ry="20" fill="#2589FF55"/><ellipse cx="125" cy="40" rx="34" ry="19" fill="#35D6ED44"/></g>
  <circle cx="215" cy="40" r="16" fill="{C['green']}" opacity=".85"/>
  <g class="umbrella"><path d="M60 140 Q130 40 200 140 Z" fill="{C['blue']}"/>
   <path d="M60 140 Q95 118 130 140 Q165 118 200 140" fill="none" stroke="{C['cyan']}" stroke-width="3"/>
   <path d="M130 140 V195 Q130 210 116 208" fill="none" stroke="{C['white']}" stroke-width="5" stroke-linecap="round"/></g>
  <g><rect class="drop" x="50" y="95" width="3" height="14" rx="1.5"/>
   <rect class="drop" x="215" y="110" width="3" height="14" rx="1.5" style="animation-delay:.5s"/>
   <rect class="drop" x="35" y="130" width="3" height="14" rx="1.5" style="animation-delay:1s"/>
   <rect class="drop" x="230" y="80" width="3" height="14" rx="1.5" style="animation-delay:.2s"/>
   <rect class="drop" x="165" y="60" width="3" height="14" rx="1.5" style="animation-delay:.8s"/></g>
 </svg>
 <span class="badge">MACHINE LEARNING PROJECT | GAUSSIAN NAIVE BAYES</span>
 <div class="hero-title">SKYCAST AI</div>
 <div class="hero-sub"><i>Predict Tomorrow. Prepare Today.</i></div>
 <p>An intelligent weather prediction system powered by Gaussian Naive Bayes Machine Learning.</p>
</div>
"""

SUN = f"""<svg width="100" height="100" viewBox="0 0 100 100"><g class="sunray">
<circle cx="50" cy="50" r="18" fill="{C['green']}"/>""" + "".join(
    f'<line x1="50" y1="14" x2="50" y2="24" stroke="{C["green"]}" stroke-width="4" stroke-linecap="round" transform="rotate({a} 50 50)"/>'
    for a in range(0, 360, 45)) + "</g></svg>"
RAIN = f"""<svg width="100" height="100" viewBox="0 0 100 100"><g class="umbrella" style="transform-origin:50px 60px">
<path d="M12 56 Q50 8 88 56 Z" fill="{C['blue']}"/><path d="M50 56 V82 Q50 92 42 90" fill="none" stroke="{C['white']}" stroke-width="4" stroke-linecap="round"/></g>
<rect class="drop" x="20" y="60" width="3" height="10" rx="1.5"/><rect class="drop" x="78" y="62" width="3" height="10" rx="1.5" style="animation-delay:.6s"/></svg>"""


def style_fig(fig, h=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=50, b=10), font=dict(family="Inter"),
                      colorway=[C["cyan"], C["green"], C["blue"], "#9ad0ff"])
    return fig


def footer():
    links = " | ".join(f"[{n}]({u})" for n, u in (("GitHub", GITHUB_URL), ("LinkedIn", LINKEDIN_URL)) if u)
    st.markdown(f'<div class="foot">SKYCAST AI is an educational Machine Learning project, not an official weather service. {links}</div>',
                unsafe_allow_html=True)


# ---------------------------- pages ----------------------------
def page_overview(train, test_n, acc, pred_rain):
    st.markdown(HERO, unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Training records", f"{len(train):,}")
    b.metric("Test records", f"{test_n:,}")
    c.metric("Model accuracy", f"{acc:.1%}")
    d.metric("Predicted rain tomorrow", f"{pred_rain:,}", help="Test rows the model labels as rainy.")
    st.subheader("What this project does")
    st.write("The system learns patterns from historical weather observations (temperature, humidity, pressure, wind, "
             "cloud and more) and classifies whether rain is expected the following day. Every number on this page is "
             "computed from your own train and test files and the trained model.")


def page_predict(art):
    cfg, model = art["cfg"], art["model"]
    st.header("Rain prediction")
    st.caption("Enter today's observations. Defaults are the training-set medians.")
    groups = {"Temperature": ["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"],
              "Rainfall and sun": ["Rainfall", "Evaporation", "Sunshine"],
              "Wind": ["WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"],
              "Humidity": ["Humidity9am", "Humidity3pm"], "Pressure": ["Pressure9am", "Pressure3pm"],
              "Cloud conditions": ["Cloud9am", "Cloud3pm"]}
    vals = {}
    with st.expander("Place and date", expanded=True):
        c1, c2, c3 = st.columns(3)
        vals["Location"] = c1.selectbox("Location", cfg["categories"]["Location"],
                                        index=cfg["categories"]["Location"].index(cfg["modes"]["Location"]), key="in_Location")
        vals["Month"] = MONTHS.index(c2.selectbox("Month", MONTHS, index=int(cfg["month_median"]) - 1, key="in_Month")) + 1
        vals["RainToday"] = c3.radio("Rain today", ["No", "Yes"], horizontal=True, key="in_RainToday")
    for g, cols in groups.items():
        with st.expander(g):
            ui = st.columns(len(cols))
            for col, name in zip(ui, cols):
                lo, hi, med = cfg["mins"][name], cfg["maxs"][name], cfg["medians"][name]
                vals[name] = col.number_input(name, min_value=float(lo), max_value=float(hi), value=float(med),
                                              help=TIPS.get(name, f"Observed range {lo:.1f} to {hi:.1f}."), key=f"in_{name}")
    with st.expander("Wind direction"):
        w = st.columns(3)
        for col, name in zip(w, ["WindGustDir", "WindDir9am", "WindDir3pm"]):
            cats = cfg["categories"][name]
            vals[name] = col.selectbox(name, cats, index=cats.index(cfg["modes"][name]), key=f"in_{name}")

    def reset():
        for k in [k for k in st.session_state if k.startswith("in_")]:
            del st.session_state[k]
        st.session_state.pop("last", None)

    b1, b2, _ = st.columns([2, 1, 3])
    go_btn = b1.button("PREDICT TOMORROW'S WEATHER", use_container_width=True)
    b2.button("Reset inputs", on_click=reset, use_container_width=True)
    if go_btn:
        with st.spinner("Running the model..."):
            row = pd.DataFrame([vals])
            p_rain = float(predict_proba_rain(model, preprocess(row, cfg))[0])
            st.session_state["last"] = (vals, p_rain)
    if "last" in st.session_state:
        vals, p_rain = st.session_state["last"]
        rain = p_rain >= 0.5
        label = "RAIN EXPECTED TOMORROW" if rain else "NO RAIN EXPECTED TOMORROW"
        tip = "Carry an umbrella or a rain jacket." if rain else "Rain looks unlikely, so a light outfit should be fine."
        st.markdown(f'<div class="result {"rain" if rain else "sun"}">{RAIN if rain else SUN}'
                    f'<div><h2>{label}</h2><div style="opacity:.85">{tip}</div></div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Model probability: rain", f"{p_rain:.1%}")
        c2.metric("Model probability: no rain", f"{1 - p_rain:.1%}")
        st.warning("This is a model prediction, not a guaranteed forecast. Probabilities are the model's own "
                   "estimates and are not calibrated real-world chances of rain.")
        report = "SKYCAST AI - Prediction report\n" + "\n".join(f"{k}: {v}" for k, v in vals.items()) + \
                 f"\n\nResult: {label}\nProbability rain: {p_rain:.4f}\nProbability no rain: {1 - p_rain:.4f}\n" \
                 "Model: Gaussian Naive Bayes (educational project; not an official forecast)\n"
        st.download_button("Download prediction report", report, "skycast_report.txt")


def page_analytics(train):
    st.header("Weather analytics")
    df = train.dropna(subset=["RainTomorrow"]).copy()
    f1, f2, f3 = st.columns(3)
    months = f1.multiselect("Month", list(range(1, 13)), format_func=lambda m: MONTHS[m - 1])
    locs = f2.multiselect("Location", sorted(df["Location"].unique()))
    cls = f3.multiselect("Rain tomorrow", ["Yes", "No"], default=["Yes", "No"])
    if months: df = df[df["Month"].isin(months)]
    if locs: df = df[df["Location"].isin(locs)]
    df = df[df["RainTomorrow"].isin(cls)]
    if df.empty:
        st.info("No rows match these filters. Clear a filter to see charts.")
        return
    cmap = {"No": C["cyan"], "Yes": C["green"]}
    st.caption(f"{len(df):,} records shown")

    cnt = df["RainTomorrow"].value_counts().rename_axis("RainTomorrow").reset_index(name="Count")
    cnt["Share"] = (cnt["Count"] / cnt["Count"].sum() * 100).round(1)
    fig = px.bar(cnt, x="RainTomorrow", y="Count", color="RainTomorrow", color_discrete_map=cmap,
                 text=cnt.apply(lambda r: f"{r.Count:,} ({r.Share}%)", axis=1), title="A. Rain tomorrow distribution")
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    c1, c2 = st.columns(2)
    fig = px.box(df[df["Rainfall"] < df["Rainfall"].quantile(.99)], x="RainTomorrow", y="Rainfall", color="RainTomorrow",
                 color_discrete_map=cmap, title="B. Rainfall by next-day outcome (top 1% hidden)")
    c1.plotly_chart(style_fig(fig), use_container_width=True)
    fig = px.histogram(df, x="Humidity3pm", color="RainTomorrow", barmode="overlay", opacity=.65, nbins=40,
                       color_discrete_map=cmap, title="C. Humidity at 3pm vs rain tomorrow")
    c2.plotly_chart(style_fig(fig), use_container_width=True)

    t = df.melt(id_vars="RainTomorrow", value_vars=["MinTemp", "MaxTemp"], var_name="Measure", value_name="Temp")
    fig = px.violin(t, x="Measure", y="Temp", color="RainTomorrow", box=True, color_discrete_map=cmap,
                    title="D. Minimum and maximum temperature by class")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    m = df.groupby("Month")["RainTomorrow"].agg(Rainy=lambda s: (s == "Yes").sum(), Total="count").reset_index()
    m["Rain rate %"] = (m["Rainy"] / m["Total"] * 100).round(1)
    m["Month name"] = m["Month"].apply(lambda x: MONTHS[int(x) - 1])
    fig = px.bar(m, x="Month name", y="Rain rate %", hover_data=["Rainy", "Total"], title="E. Share of days followed by rain, by month")
    fig.update_traces(marker_color=C["blue"])
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    w = df.dropna(subset=["WindGustDir"]).groupby(["WindGustDir", "RainTomorrow"]).size().reset_index(name="Count")
    fig = px.bar(w, x="WindGustDir", y="Count", color="RainTomorrow", barmode="group", color_discrete_map=cmap,
                 title="F. Rain outcome by wind gust direction")
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    num = df[NUM].copy()
    num["RainTomorrow (1=Yes)"] = (df["RainTomorrow"] == "Yes").astype(int)
    corr = num.corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale=[C["navy"], C["blue"], C["cyan"], C["green"]],
                    title="G. Feature correlation heatmap")
    st.plotly_chart(style_fig(fig, 640), use_container_width=True)


def page_performance(art):
    st.header("Model performance")
    y, p, n = evaluate_test()
    m = metric_dict(y, p)
    for col, (k, v) in zip(st.columns(4), m.items()):
        col.metric(k, f"{v:.3f}")
    st.caption(f"Computed on {n:,} labelled test rows. Precision, recall and F1 describe the rain class.")
    cm = confusion_matrix(y, p, labels=[0, 1])
    fig = px.imshow(cm, text_auto=",", x=["Pred: No rain", "Pred: Rain"], y=["Actual: No rain", "Actual: Rain"],
                    color_continuous_scale=[C["navy"], C["blue"], C["cyan"]], title="Confusion matrix")
    c1, c2 = st.columns(2)
    c1.plotly_chart(style_fig(fig), use_container_width=True)
    comp = pd.DataFrame({"Label": ["No rain", "Rain"] * 2, "Source": ["Actual"] * 2 + ["Predicted"] * 2,
                         "Count": [int((y == 0).sum()), int((y == 1).sum()), int((p == 0).sum()), int((p == 1).sum())]})
    fig = px.bar(comp, x="Label", y="Count", color="Source", barmode="group", title="Actual vs predicted labels",
                 color_discrete_sequence=[C["cyan"], C["green"]])
    c2.plotly_chart(style_fig(fig), use_container_width=True)
    st.subheader("Classification report")
    rep = pd.DataFrame(classification_report(y, p, target_names=["No rain", "Rain"], output_dict=True, zero_division=0)).T
    st.dataframe(rep.round(3), use_container_width=True)
    st.subheader("How it works")
    st.write("Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes "
             "conditional independence between features given the target class and models numerical features using "
             "Gaussian distributions.")
    st.markdown("**Weather Features → Data Preprocessing → Gaussian Naive Bayes → Class Probabilities → Rain Prediction**")
    st.markdown("- **Accuracy:** share of all predictions that were right.\n"
                "- **Precision:** when the model says rain, how often it is right.\n"
                "- **Recall:** of the days that really had rain, how many the model caught.\n"
                "- **F1-score:** one balanced number combining precision and recall.")
    if m["Precision"] < 0.5:
        st.info("Rain precision is below 50%, so the model raises many false alarms. Treat it as a learning "
                "experiment rather than a dependable forecaster.")


def page_batch(art):
    cfg, model = art["cfg"], art["model"]
    st.header("Upload your weather data, get instant predictions")
    st.write("Upload a CSV with these columns: " + ", ".join(REQUIRED) + ", and either **Date** or **Month**. "
             "`RainTomorrow` is optional and is only used for evaluation, never as an input.")
    up = st.file_uploader("Weather CSV", type="csv")
    if up is None:
        st.info("Choose a CSV file to start.")
        return
    try:
        df = pd.read_csv(up)
    except Exception as e:
        st.error(f"This file could not be read as a CSV: {e}")
        return
    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)
    missing = [c for c in REQUIRED if c not in df.columns]
    if "Date" not in df.columns and "Month" not in df.columns:
        missing.append("Date or Month")
    if missing:
        st.error("These required columns are missing: " + ", ".join(missing) + ". Add them and upload again.")
        return
    with st.spinner("Predicting..."):
        X = preprocess(df, cfg)
        proba = predict_proba_rain(model, X)
        out = df.copy()
        out["Predicted_RainTomorrow"] = np.where(proba >= 0.5, "Yes", "No")
        out["Probability_No_Rain"] = (1 - proba).round(4)
        out["Probability_Rain"] = proba.round(4)
    unseen = {c: sorted(set(df[c].dropna().astype(str)) - set(cfg["categories"][c])) for c in CAT}
    for c, u in unseen.items():
        if u:
            st.warning(f"{c} has values the model has never seen ({', '.join(u[:5])}). They were replaced by the training mode.")
    rainy = int((proba >= 0.5).sum())
    a, b, c = st.columns(3)
    a.metric("Rows predicted", f"{len(out):,}")
    b.metric("Rain expected", f"{rainy:,}")
    c.metric("No rain expected", f"{len(out) - rainy:,}")
    st.success("Predictions are ready.")
    st.dataframe(out.head(200), use_container_width=True)
    st.download_button("Download results CSV", out.to_csv(index=False).encode(), "skycast_predictions.csv", "text/csv")
    if "RainTomorrow" in df.columns:
        y = target_series(df["RainTomorrow"])
        mask = y.notna()
        if mask.sum() == 0:
            st.warning("RainTomorrow has no usable Yes/No labels, so evaluation was skipped.")
        else:
            st.subheader("Evaluation on your file")
            yt, yp = y[mask].astype(int), (proba[mask.to_numpy()] >= 0.5).astype(int)
            for col, (k, v) in zip(st.columns(4), metric_dict(yt, yp).items()):
                col.metric(k, f"{v:.3f}")
            st.caption(f"{int(mask.sum()):,} rows with a known label were used; blank labels were excluded.")


def page_about():
    st.header("About the project")
    st.markdown("**Project name:** SKYCAST AI - Weather Intelligence  \n**Algorithm:** Gaussian Naive Bayes  \n"
                "**Problem type:** Binary classification  \n**Target variable:** RainTomorrow  \n"
                "**Dataset:** Historical weather observations (your train_80.csv and test_20.csv)  \n"
                "**Framework:** Streamlit  \n**Visualization:** Plotly")
    st.subheader("Why Naive Bayes")
    st.write("Naive Bayes is fast, simple and outputs class probabilities, which makes it a good first probabilistic "
             "classifier. Its key assumption is that features are independent given the class. Weather variables "
             "are strongly related (humidity, cloud and pressure move together), so the assumption is only "
             "approximately true and limits accuracy.")
    st.subheader("Workflow")
    st.markdown("Raw CSV → fill missing values (training medians and modes) → Date to Month → Yes/No to 1/0 → "
                "one-hot encoding → Gaussian Naive Bayes → probabilities → prediction")
    st.subheader("Limitations")
    st.markdown("- Next-day rain depends on atmospheric dynamics that a single day's observations cannot fully capture.\n"
                "- About 22% of days are rainy, so the classes are imbalanced.\n"
                "- Medians fill large gaps in Sunshine, Evaporation and Cloud columns, which can blur their signal.\n"
                "- Results hold for the stations and years in the dataset only.")
    if GITHUB_URL or LINKEDIN_URL:
        st.markdown(" | ".join(f"[{n}]({u})" for n, u in (("GitHub", GITHUB_URL), ("LinkedIn", LINKEDIN_URL)) if u))
    else:
        st.caption("Set GITHUB_URL and LINKEDIN_URL at the top of dashboard.py to show your links here.")


# ---------------------------- app ----------------------------
def main():
    st.set_page_config(page_title="SKYCAST AI", page_icon="☂️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    for f in (TRAIN_CSV, TEST_CSV):
        if not f.exists():
            st.error(f"File not found: {f.name}. Place train_80.csv and test_20.csv in the same folder as dashboard.py.")
            st.stop()
    art = load_artifacts()
    train = load_raw(TRAIN_CSV)
    nav = {"🏠  Overview": "o", "🌧️  Rain Prediction": "p", "📊  Weather Analytics": "a",
           "🎯  Model Performance": "m", "📁  Batch Predictions": "b", "ℹ️  About the Project": "i"}
    with st.sidebar:
        st.markdown("### SKYCAST AI")
        choice = nav[st.radio("Navigate", list(nav), label_visibility="collapsed")]
        st.caption("Gaussian Naive Bayes | educational project")
        if st.button("Retrain model"):
            ARTIFACT.unlink(missing_ok=True)
            st.cache_resource.clear(); st.cache_data.clear()
            st.rerun()
    if choice == "o":
        y, p, n = evaluate_test()
        page_overview(train, n, accuracy_score(y, p), int(p.sum()))
    elif choice == "p": page_predict(art)
    elif choice == "a": page_analytics(train)
    elif choice == "m": page_performance(art)
    elif choice == "b": page_batch(art)
    else: page_about()
    footer()


main()