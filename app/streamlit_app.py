"""NHS A&E Demand Forecast — Streamlit dashboard.

Run locally:  streamlit run app/streamlit_app.py
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PROC = ROOT / "data" / "processed"

st.set_page_config(page_title="NHS A&E Demand Forecast", page_icon="🏥", layout="wide")

MEASURES = {"Type 1 A&E attendances": "attendances", "Emergency admissions via Type 1 A&E": "emergency_admissions"}
LEVEL_ORDER = ["England", "Region", "ICB", "Trust"]
MODEL_NAMES = {"snaive": "Seasonal naive", "ets": "ETS (Holt-Winters)", "sarima": "SARIMA airline",
               "lgbm": "Global LightGBM", "combo": "Ensemble (ETS+SARIMA+LGBM)"}
RECON_NAMES = {"base": "None (base)", "bu": "Bottom-up", "td": "Top-down", "ols": "OLS",
               "wls_struct": "WLS (structural)", "mint_shrink": "MinT (shrink)"}


@st.cache_data
def load(measure):
    fc = pd.read_csv(RES / f"forecast_{measure}.csv", parse_dates=["month"])
    hist = pd.read_csv(RES / f"history_{measure}.csv", parse_dates=["month"])
    covid = fc["covid"].iloc[0]
    summ = pd.read_csv(RES / f"backtest_summary_{measure}.csv").query("covid == @covid")
    hz = pd.read_csv(RES / f"backtest_by_horizon_{measure}.csv").query("covid == @covid")
    return fc, hist, summ, hz


@st.cache_data
def load_hier():
    return pd.read_csv(PROC / "hierarchy.csv")


st.title("🏥 NHS A&E Demand Forecast")
st.caption("Coherent 12-month forecasts of Type 1 A&E demand for 120 acute trusts in England, plus their 36 ICBs, "
           "7 regions and the national total. The forecasts are reconciled so every level adds up. "
           "Source: NHS England Monthly A&E Attendances & Emergency Admissions, Apr 2018 – latest month.")

with st.sidebar:
    measure_label = st.radio("Measure", list(MEASURES))
    measure = MEASURES[measure_label]
    hier = load_hier()
    region = st.selectbox("Region", ["All England"] + sorted(hier.region_name.unique()))
    sub = hier if region == "All England" else hier[hier.region_name == region]
    icb_opts = sub.drop_duplicates("icb_code").sort_values("icb_name")
    icb = st.selectbox("ICB", ["All ICBs in selection"] + icb_opts.icb_name.tolist())
    sub2 = sub if icb.startswith("All") else sub[sub.icb_name == icb]
    trust = st.selectbox("Trust", ["All trusts in selection"] + sorted(sub2.trust_name.tolist()))

fc, hist, summ, hz = load(measure)
model, recon = fc["model"].iloc[0], fc["recon"].iloc[0]

if not trust.startswith("All"):
    sid = "T:" + hier.loc[hier.trust_name == trust, "trust_code"].iloc[0]
elif not icb.startswith("All"):
    sid = "I:" + icb_opts.loc[icb_opts.icb_name == icb, "icb_code"].iloc[0]
elif region != "All England":
    sid = f"R:{region}"
else:
    sid = "England"

f = fc[fc.series == sid]
h = hist[hist.series == sid]
name = f["name"].iloc[0]

tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "❄️ Winter outlook", "🧪 Model comparison", "🔗 How reconciliation works"])

with tab1:
    last_year = h.set_index("month")["actual"]
    nxt12 = f["forecast"].sum()
    prev12 = last_year.iloc[-12:].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected series", name if len(name) < 40 else name[:38] + "…")
    c2.metric("Next 12 months (forecast)", f"{nxt12:,.0f}", f"{(nxt12 / prev12 - 1):+.1%} vs last 12 months")
    peak = f.loc[f.forecast.idxmax()]
    c3.metric("Busiest forecast month", peak.month.strftime("%b %Y"), f"{peak.forecast:,.0f}")
    c4.metric("Model used", MODEL_NAMES[model], RECON_NAMES[recon] + " reconciliation", delta_color="off")

    fig = go.Figure()
    fig.add_vrect(x0="2020-03-15", x1="2021-06-15", fillcolor="grey", opacity=0.12, line_width=0,
                  annotation_text="COVID period (adjusted in training)", annotation_position="top left")
    fig.add_trace(go.Scatter(x=h.month, y=h.actual, name="Actual", line=dict(color="#1f3b73", width=2)))
    fig.add_trace(go.Scatter(x=pd.concat([f.month, f.month[::-1]]), y=pd.concat([f.hi95, f.lo95[::-1]]),
                             fill="toself", fillcolor="rgba(0,114,206,0.12)", line=dict(width=0), name="95% interval"))
    fig.add_trace(go.Scatter(x=pd.concat([f.month, f.month[::-1]]), y=pd.concat([f.hi80, f.lo80[::-1]]),
                             fill="toself", fillcolor="rgba(0,114,206,0.25)", line=dict(width=0), name="80% interval"))
    fig.add_trace(go.Scatter(x=f.month, y=f.forecast, name="Forecast", line=dict(color="#0072CE", width=2.5, dash="dash")))
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=measure_label,
                      legend=dict(orientation="h", y=-0.15), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    show = f[["month", "forecast", "lo80", "hi80", "lo95", "hi95"]].copy()
    show["month"] = show.month.dt.strftime("%b %Y")
    st.dataframe(show.set_index("month").round(0).astype(int), width="stretch")

with tab2:
    st.subheader("Winter outlook: December to February")
    st.caption("Forecast winter demand per trust compared with last winter. Trusts facing the biggest rise "
               "are where escalation planning should start.")
    wm = fc[fc.month.dt.month.isin([12, 1, 2])]
    wm = wm[wm.month <= wm.month.min() + pd.DateOffset(months=2)]  # the next winter only
    ly_months = [m - pd.DateOffset(years=1) for m in wm.month.unique()]
    trusts_f = wm[wm.level == "Trust"].groupby(["series", "name"]).forecast.sum()
    trusts_ly = hist[hist.month.isin(ly_months) & hist.series.str.startswith("T:")].groupby("series").actual.sum()
    w = trusts_f.reset_index().merge(trusts_ly.rename("last_winter").reset_index(), on="series")
    w["change"] = w.forecast / w.last_winter - 1
    w = w.merge(hier.assign(series="T:" + hier.trust_code)[["series", "region_name", "icb_name"]], on="series")
    if region != "All England":
        w = w[w.region_name == region]
    w = w.sort_values("change", ascending=False)
    disp = w[["name", "icb_name", "region_name", "last_winter", "forecast", "change"]].rename(columns={
        "name": "Trust", "icb_name": "ICB", "region_name": "Region", "last_winter": "Last winter (actual)",
        "forecast": "This winter (forecast)", "change": "Change"})
    st.dataframe(disp.style.format({"Last winter (actual)": "{:,.0f}", "This winter (forecast)": "{:,.0f}",
                                    "Change": "{:+.1%}"}).background_gradient(subset=["Change"], cmap="RdYlGn_r"),
                 width="stretch", height=520)

with tab3:
    covid_txt = {"post": "trained on post-COVID data only (Jul 2021 onwards)",
                 "impute": "trained on full history with COVID months replaced by a counterfactual"}[fc["covid"].iloc[0]]
    st.subheader("Rolling-origin backtest (9 forecast origins, Oct 2023 – Oct 2025, 12 months ahead)")
    st.caption(f"Models shown are {covid_txt}: the COVID treatment that won the backtest for this measure.")
    st.caption("MASE below 1 means the method beats a seasonal-naive forecast. CRPS scores the whole forecast "
               "distribution, not just the point forecast. Lower is better for both.")
    metric = st.radio("Metric", ["MASE", "CRPS", "MAPE", "cov80", "cov95"], horizontal=True)
    piv = summ.pivot_table(index=["model", "recon"], columns="level", values=metric)[LEVEL_ORDER]
    piv["Average"] = piv.mean(axis=1)
    piv = piv.sort_values("Average", ascending=metric not in ["cov80", "cov95"])
    piv.index = [f"{MODEL_NAMES[m]} · {RECON_NAMES[r]}" for m, r in piv.index]
    fmt = "{:.1%}" if metric in ["MAPE", "cov80", "cov95"] else "{:.3f}"
    st.dataframe(piv.style.format(fmt).background_gradient(cmap="RdYlGn_r" if metric not in ["cov80", "cov95"] else "Blues"),
                 width="stretch", height=560)

    st.markdown("**How accuracy changes with the forecast horizon** (chosen configuration, MAPE)")
    hzs = hz[(hz.model == model) & (hz.recon == recon)]
    fig2 = go.Figure()
    for lvl in LEVEL_ORDER:
        d = hzs[hzs.level == lvl]
        fig2.add_trace(go.Scatter(x=d.h, y=d.MAPE, name=lvl, mode="lines+markers"))
    fig2.update_layout(height=360, xaxis_title="Months ahead", yaxis_title="MAPE", yaxis_tickformat=".0%",
                       margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, width="stretch")

with tab4:
    st.markdown("""
**The problem.** If you forecast each trust, each ICB, each region and England separately, the numbers won't add up.
The sum of the trust forecasts won't equal the England forecast. Planners at different levels end up working from different versions.

**The fix.** Forecast all 164 series, then *reconcile* them: adjust them as little as possible so they become coherent.

* **Bottom-up**: forecast the trusts and add them up. Coherent, but it ignores the cleaner signal in the aggregate series.
* **Top-down**: forecast England and split it using historical shares. It ignores what is happening at each trust.
* **OLS / WLS**: least-squares projection onto the coherent space. WLS gives each series a weight based on how many trusts sit under it.
* **MinT (minimum trace)**: uses the covariance of the base-forecast errors so that noisier series are adjusted more.
  This is the optimal linear reconciliation (Wickramasuriya, Athanasopoulos & Hyndman, 2019).

The backtest tab shows which approach wins on this data. Every forecast in this dashboard is coherent: trusts add up to ICBs, ICBs to regions, and regions to England.
""")
    st.markdown("**Hierarchy:** 120 trusts → 36 ICBs → 7 regions → England = 164 series.")
