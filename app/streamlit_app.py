"""NHS A&E Demand Forecast — Streamlit dashboard.

Run locally:  streamlit run app/streamlit_app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

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


@st.cache_data
def load_four_hour():
    lv = pd.read_csv(RES / "four_hour_forecast.csv", parse_dates=["target_month"])
    perf = pd.read_csv(PROC / "trust_month_panel.csv", parse_dates=["month"])[["trust_code", "month", "pct_within_4h"]]
    summ = pd.read_csv(RES / "four_hour_backtest_summary.csv", index_col=0)
    cal = pd.read_csv(RES / "four_hour_calibration.csv")
    imp = pd.read_csv(RES / "four_hour_feature_importance.csv", index_col=0)
    return lv, perf, summ, cal, imp


Q_LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
Q_COLS = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]


def prob_below(q: np.ndarray, thr: float) -> float:
    """CDF from quantiles: linear inside, Gaussian tails outside (same rule as src/four_hour.py)."""
    sd = max((q[-1] - q[0]) / (2 * norm.ppf(0.95)), 1e-4)
    if thr <= q[0]:
        return float(norm.cdf(norm.ppf(0.05) + (thr - q[0]) / sd))
    if thr >= q[-1]:
        return float(norm.cdf(norm.ppf(0.95) + (thr - q[-1]) / sd))
    return float(np.interp(thr, q, Q_LEVELS))


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

tab1, tab2, tab5, tab3, tab4 = st.tabs(["📈 Forecast", "❄️ Winter outlook", "⏱️ 4-hour risk", "🧪 Model comparison",
                                        "🔗 How reconciliation works"])

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

    st.caption(f"Intervals: {f['interval_method'].iloc[0]}. Chosen by an online coverage test in `src/calibrate.py`.")
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

with tab5:
    lv, perf4, summ4, cal4, imp4 = load_four_hour()
    st.subheader("Type 1 A&E 4-hour performance: risk by trust")
    st.caption("Share of Type 1 (major A&E) attendances admitted, transferred or discharged within 4 hours. "
               "The national 82% objective for 2026/27 covers all A&E types, including urgent treatment centres. "
               "Type 1 performance runs far lower (about 60% nationally), so set the threshold for Type 1 below.")
    c1, c2 = st.columns([1, 2])
    thr = c1.slider("Threshold (Type 1 % within 4 hours)", 40, 90, 60, step=1) / 100
    period = c2.radio("Period", ["Next winter (Dec–Feb)", "Next 3 months", "Next 12 months"], horizontal=True)
    if period.startswith("Next winter"):
        sel = lv[lv.target_month.dt.month.isin([12, 1, 2])]
        sel = sel[sel.target_month <= sel.target_month.min() + pd.DateOffset(months=2)]
    elif period == "Next 3 months":
        sel = lv[lv.h <= 3]
    else:
        sel = lv
    sel = sel.copy()
    sel["p_below"] = [prob_below(r, thr) for r in sel[Q_COLS].to_numpy()]
    if region != "All England":
        sel = sel[sel.region_name == region]
    risk = (sel.groupby(["trust_code", "trust_name", "icb_name", "region_name"])
            .agg(median=("q50", "mean"), lo=("q10", "mean"), hi=("q90", "mean"),
                 p_below=("p_below", "mean"), p_worse=("p_worse_than_ly", "mean"))
            .reset_index().sort_values("p_below", ascending=False))
    m1, m2, m3 = st.columns(3)
    m1.metric("Trusts in selection", len(risk))
    m2.metric(f"Trusts more likely than not below {thr:.0%}", int((risk.p_below > 0.5).sum()))
    m3.metric("Median forecast performance", f"{risk['median'].median():.1%}")
    disp = risk.rename(columns={"trust_name": "Trust", "icb_name": "ICB", "region_name": "Region",
                                "median": "Forecast (median)", "lo": "80% range low", "hi": "80% range high",
                                "p_below": f"P(below {thr:.0%})", "p_worse": "P(worse than same month last year)"})
    disp = disp.drop(columns=["trust_code"])
    st.dataframe(disp.style.format({c: "{:.1%}" for c in disp.columns if c not in ["Trust", "ICB", "Region"]})
                 .background_gradient(subset=[f"P(below {thr:.0%})"], cmap="RdYlGn_r", vmin=0, vmax=1),
                 width="stretch", height=420)

    tcode = (hier.loc[hier.trust_name == trust, "trust_code"].iloc[0] if not trust.startswith("All")
             else risk.trust_code.iloc[0])
    tname = hier.loc[hier.trust_code == tcode, "trust_name"].iloc[0]
    h4 = perf4[(perf4.trust_code == tcode) & (perf4.month >= "2021-07-01")]
    f4 = lv[lv.trust_code == tcode]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=h4.month, y=h4.pct_within_4h, name="Actual", line=dict(color="#1f3b73")))
    fig4.add_trace(go.Scatter(x=pd.concat([f4.target_month, f4.target_month[::-1]]), y=pd.concat([f4.q95, f4.q05[::-1]]),
                              fill="toself", fillcolor="rgba(0,114,206,0.12)", line=dict(width=0), name="90% range"))
    fig4.add_trace(go.Scatter(x=pd.concat([f4.target_month, f4.target_month[::-1]]), y=pd.concat([f4.q90, f4.q10[::-1]]),
                              fill="toself", fillcolor="rgba(0,114,206,0.25)", line=dict(width=0), name="80% range"))
    fig4.add_trace(go.Scatter(x=f4.target_month, y=f4.q50, name="Median forecast", line=dict(color="#0072CE", dash="dash")))
    fig4.add_hline(y=thr, line_dash="dot", line_color="#DA291C", annotation_text=f"threshold {thr:.0%}")
    fig4.update_layout(height=380, title=f"{tname}: Type 1 4-hour performance", yaxis_tickformat=".0%",
                       margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig4, width="stretch")
    st.caption("Showing the trust picked in the sidebar, or else the highest-risk trust in the table.")

    with st.expander("How good is this model? (backtest, 7 forecast origins, 12 months ahead)"):
        mae = summ4["MAE (pct points)"].dropna()
        st.markdown(
            f"- **Median forecast error:** {mae.iloc[0]:.2f} percentage points, against "
            f"{mae.iloc[1:].min():.2f} for the best naive rule (last month, last 3 months or same month last year).\n"
            f"- **Event probabilities:** Brier skill vs always predicting the base rate is "
            f"{1 - summ4.loc['Brier score: below 60%', 'Brier'] / summ4.loc['Brier score: below 60%', 'Brier (climatology)']:.0%} "
            f"for *below 60%* and "
            f"{1 - summ4.loc['Brier score: worse than same month last year', 'Brier'] / summ4.loc['Brier score: worse than same month last year', 'Brier (climatology)']:.0%} "
            f"for *worse than last year*.\n"
            f"- **Range coverage after conformal calibration:** 80% range covers "
            f"{summ4.loc['Interval coverage (conformal)', '80% (q10-q90)']:.0%} and 90% range covers "
            f"{summ4.loc['Interval coverage (conformal)', '90% (q05-q95)']:.0%} (raw quantile model: "
            f"{summ4.loc['Interval coverage (raw quantiles)', '80% (q10-q90)']:.0%} / "
            f"{summ4.loc['Interval coverage (raw quantiles)', '90% (q05-q95)']:.0%}).")
        c = cal4[cal4.event == "below 60%"]
        figc = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="grey"), name="perfect"),
                          go.Scatter(x=c.predicted, y=c.observed, mode="lines+markers", name="model",
                                     text=c.n, hovertemplate="predicted %{x:.0%}<br>observed %{y:.0%}<br>n=%{text}")])
        figc.update_layout(height=300, title="Calibration: P(below 60%)", xaxis_title="predicted probability",
                           yaxis_title="observed frequency", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(figc, width="stretch")
        st.markdown("**What drives the median forecast** (share of LightGBM gain)")
        st.bar_chart(imp4["gain_share"])

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
