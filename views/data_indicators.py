"""Data & Indicators — sources, provenance, window explanation, controls and market-data charts."""
import streamlit as st

from core import charts as G
from core import config as C
from core import controls, exports, state, ui_components as ui
from core.data_gwp2 import vix_method_comparison

run = state.get_run()
ds = state.load_dataset(run)

ui.page_header("Data & Indicators", "Where every number comes from, how the monthly panel is built, and how to change the analysis window.",
               "Research overview")
ui.config_banner(run)

with st.expander("Data controls — period, VIX aggregation, data source", expanded=False):
    controls.data_controls(ds)

for w in ds.warnings:
    st.warning(w)

w = ds.window
ui.kpis([
    ("Master panel", f"{w['panel_rows']} × {ds.master.shape[1]}", f"{w['panel_first']} → {w['panel_last']}", ""),
    ("Feature matrix", f"{w['feature_rows']} × 2", f"{w['feature_first']} → {w['feature_last']}", "acc"),
    ("Warm-up months", str(w["warmup_rows"]), "lost to return + rolling window", ""),
    ("Binding series", w["binding_start_series_id"], f"first obs. {w['first_valid_in_source_window'][w['binding_start_series']]}", ""),
])

ui.section("Why the dates are what they are", "Derived from the loaded data, not hard-coded.")
ui.dual_explanation(
    f"The project asks for data from {C.START[:7]} onward, but the dollar index only begins in {w['first_valid_in_source_window']['DXY'][:7]}. "
    f"Every indicator must exist in a month for the month to be used, so the panel starts {w['panel_first']} ({w['panel_rows']} months). "
    f"Computing a return needs the previous month and a 12-month volatility needs 12 returns, so the first {w['warmup_rows']} months are only 'warm-up'. "
    f"The model therefore sees {w['feature_rows']} months, {w['feature_first']} to {w['feature_last']}.",
    f"- Source window: **{w['source_window']}**; WTI has **{w['wti_months_in_source_window']}** monthly observations in it.\n"
    f"- First valid observation inside the window by series: " + ", ".join(f"`{k}` {v}" for k, v in w["first_valid_in_source_window"].items()) + ".\n"
    f"- Binding series: **{w['binding_start_series_id']}** (inner join over all five series).\n"
    f"- Panel: {w['panel_rows']} rows ({w['panel_first']}–{w['panel_last']}). Warm-up: {w['warmup_explained']}\n"
    f"- Modelled sample: {w['feature_rows']} rows ({w['feature_first']}–{w['feature_last']}).\n"
    f"- Selectable modelled range with the current data: {w['available_feature_range']}.\n\n"
    "**Report note.** The submitted report text says '216 months from January 2006'. 216 months ending December 2024 start in "
    "**January 2007**; January 2006 is the start of the 228-row *panel*. The data and code are consistent with each other; the report wording is imprecise.")

ui.section("Data-source panel")
st.caption("‘Latest observation’ is the last **historical** value in the file, not a current market quote. "
           "‘Bundled snapshot’ values come from files stored with the app and are never described as live.")
cols = ["Indicator", "Provider", "Series ID", "Frequency", "Units", "Start date", "End date", "Observations",
        "Missing values", "Months forward-filled", "Latest observation", "Last retrieval", "Origin", "Status"]
st.dataframe(ds.status[cols], width="stretch", hide_index=True)
with st.expander("Descriptions, monthly rules and notes"):
    for _, r in ds.status.iterrows():
        spec = C.SERIES[r["Key"]]
        st.markdown(f"**{r['Indicator']}** — {r['Description']}. *Monthly rule:* {spec.monthly_rule}. *Role:* {spec.role}. *Notes:* {r['Notes']}")
st.caption("‘Missing values’ counts placeholder rows (FRED uses ‘.’) removed from the raw file. "
           "‘Months forward-filled’ counts months inside the modelled panel that had no observation.")

ui.section("VIX: month-end close or monthly mean?")
cmp = vix_method_comparison(ds)
if cmp is None:
    st.info("Only monthly VIX bars are available from the active provider, so the two aggregations cannot be compared.")
else:
    a, b, c = st.columns(3)
    a.metric("Correlation", f"{cmp['Month-end close'].corr(cmp['Monthly mean']):.4f}")
    b.metric("Mean absolute difference", f"{cmp['Difference (close - mean)'].abs().mean():.2f} pts")
    big = cmp["Difference (close - mean)"].abs().idxmax()
    c.metric("Largest gap", f"{cmp.loc[big, 'Difference (close - mean)']:+.1f} pts", f"{big:%b %Y}")
    ui.callout("note", "Which one is used",
               f"The report describes Yahoo Finance **monthly closes**. FRED's daily VIX close resampled to the last trading day of the month is the same quantity. "
               f"Active setting: **{ds.vix_aggregation_effective.replace('_', ' ')}**. The earlier app version used monthly means; "
               "that is still selectable as an alternative. The HMM features do not use VIX — the choice only affects VIX-based regime profiles and the Bayesian Network.")
    st.line_chart(cmp[["Month-end close", "Monthly mean"]], color=[C.THEME["accent"], C.THEME["info"]])

ui.section("Market data charts")
sel = st.multiselect("Indicators shown in the comparison chart", list(C.SERIES), default=list(C.SERIES),
                     format_func=lambda k: C.SERIES[k].label)
t1, t2, t3, t4 = st.tabs(["WTI price", "Returns", "Volatility", "Macro comparison"])
with t1:
    st.plotly_chart(G.fig_price(ds.master), width="stretch")
with t2:
    st.plotly_chart(G.fig_returns(ds.features), width="stretch")
with t3:
    st.plotly_chart(G.fig_vol(ds.features), width="stretch")
with t4:
    if sel:
        st.plotly_chart(G.fig_macro(ds.master[sel]), width="stretch")
        st.caption("All series except VIX are indexed to 100 at the start of the panel so scales are comparable.")
    else:
        st.info("Select at least one indicator.")

ui.section("Tables and downloads")
tab_m, tab_f = st.tabs(["Master panel", "Feature matrix"])
with tab_m:
    st.dataframe(ds.master.round(3), width="stretch")
    st.download_button("Download master panel (CSV)", exports.master_csv(ds), "master_panel.csv", "text/csv")
with tab_f:
    st.dataframe(ds.features.round(5), width="stretch")
    st.download_button("Download feature matrix (CSV)", exports.features_csv(ds), "feature_matrix.csv", "text/csv")
st.download_button("Download data-source table (CSV)", exports.to_csv_bytes(ds.status, index=False), "data_sources.csv", "text/csv")
ui.disclaimer()
