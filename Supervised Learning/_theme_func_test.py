import ast
import io
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

src = Path("NaiveBayesDashboard.py").read_text(encoding="utf-8")
ast.parse(src)
print("syntax: ok")

import NaiveBayesDashboard as d

assert set(d.THEMES) == {"dark", "light"}
for name, t in d.THEMES.items():
    for k in ["bg", "sidebar", "card", "elev", "plot", "blue", "cyan", "green", "text", "muted", "border"]:
        assert k in t, (name, k)
print("themes: ok")

art = d.load_artifacts()
y, pred, n = d.evaluate_test()
m = d.metric_dict(y, pred)
print("eval", n, {k: round(v, 4) for k, v in m.items()}, "pred_rain", int(pred.sum()))

cfg, model = art["cfg"], art["model"]
row = {c: cfg["medians"][c] for c in d.NUM}
row["RainToday"] = "No"
row["Month"] = int(cfg["month_median"])
for c in d.CAT:
    row[c] = cfg["modes"][c]
X = d.preprocess(pd.DataFrame([row]), cfg)
pr = float(d.predict_proba_rain(model, X)[0])
print("single_pred", round(pr, 4), "label", "Rain" if pr >= 0.5 else "No rain")

test = d.load_raw(d.TEST_CSV).head(5)
out, proba = d.batch_predict(test, cfg, model)
assert "Predicted_RainTomorrow" in out.columns
print("batch_ok", len(out), out["Predicted_RainTomorrow"].tolist())

try:
    pd.read_csv(io.StringIO("%%%not csv%%%\n\n\n"))
except Exception as e:
    print("bad_csv:", type(e).__name__)

try:
    d.batch_predict(pd.DataFrame({"foo": [1]}), cfg, model)
    raise SystemExit("missing_cols: UNEXPECTED SUCCESS")
except ValueError as e:
    print("missing_cols:", str(e)[:90])

print("download_bytes", len(out.to_csv(index=False).encode()))

for key in ("dark", "light"):
    t = d.THEMES[key]
    fig = go.Figure(go.Bar(x=["a", "b"], y=[1, 2], name="demo"))
    fig.update_layout(title="Theme check")
    styled = d.style_fig(fig, t, 200)
    assert styled.layout.paper_bgcolor == t["plot"]
    assert styled.layout.font.color == t["muted"]
print("plotly_theme: ok")
print("ALL FUNCTIONAL CHECKS PASSED")
