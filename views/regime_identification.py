"""Market Regime Identification — transparent rule-based Bull / Bear / Stagnant benchmark (Step 2)."""
import pandas as pd
import streamlit as st

from core import charts as G
from core import config as C
from core import state, ui_components as ui
from core.hmm_engine import compute_spells, regime_summary, rule_based_regimes

run = state.get_run()
ds = state.load_dataset(run)

ui.page_header("Market Regime Identification", "A simple, fully transparent classification of every month — the yardstick against which the HMM is later judged.",
               "HMM research · Step 2")
ui.callout("warn", "This is not the HMM",
           "These labels come from two fixed rules on trailing returns and volatility. The HMM-generated regimes (Regime Detection Pipeline) are learned from the data and can differ. "
           "The two are compared, never confused.")

ui.section("Classification rules")
st.dataframe(pd.DataFrame({
    "Regime": ["Bull", "Bear", "Stagnant"],
    "Rule": ["Trailing 12-month WTI return > +10% AND annualised volatility < 40%",
             "Trailing 12-month WTI return < −10% AND annualised volatility > 30%",
             "All remaining months"],
    "Intuition": ["Sustained rise without extreme turbulence", "Sustained fall with elevated turbulence", "No clear signal"]}),
    width="stretch", hide_index=True)

with st.expander("Explore other thresholds (experimental — not the report's rules)"):
    c1, c2, c3, c4 = st.columns(4)
    br = c1.slider("Bull: min 12M return (%)", 0, 40, 10, key="rb_br")
    bv = c2.slider("Bull: max volatility (%)", 10, 80, 40, key="rb_bv")
    er = c3.slider("Bear: max 12M return (%)", -40, 0, -10, key="rb_er")
    ev = c4.slider("Bear: min volatility (%)", 10, 80, 30, key="rb_ev")
default_thr = (br, bv, er, ev) == (10, 40, -10, 30)
if not default_thr:
    ui.badges([("Alternative thresholds — not the submitted report", "warn")])

reg_full = rule_based_regimes(ds.master, br / 100, bv / 100, er / 100, ev / 100)
reg = reg_full.loc[ds.features.index]
summ = regime_summary(ds.master, reg)
counts = reg.value_counts()

ui.kpis([("Bull months", str(int(counts.get("Bull", 0))), f"{counts.get('Bull', 0) / len(reg):.1%} of {len(reg)}", "bull"),
         ("Stagnant months", str(int(counts.get("Stagnant", 0))), f"{counts.get('Stagnant', 0) / len(reg):.1%}", "stag"),
         ("Bear months", str(int(counts.get("Bear", 0))), f"{counts.get('Bear', 0) / len(reg):.1%}", "bear"),
         ("Sample", f"{len(reg)} months", ds.fmt_range("features"), "")])
if default_thr and ds.is_report_window:
    ref = C.REPORT_REFERENCE["rule_based_counts"]
    same = all(int(counts.get(k, 0)) == v for k, v in ref.items())
    st.caption("Report reference counts — Bull {Bull}, Bear {Bear}, Stagnant {Stagnant}: ".format(**ref) + ("**match**." if same else "**differ** (see Methodology → discrepancy register)."))

ui.section("Historical behaviour")
st.plotly_chart(G.fig_regime_timeline(reg, ["Bear", "Stagnant", "Bull"], "Rule-based regime timeline"), width="stretch")
st.plotly_chart(G.fig_price(ds.master.loc[ds.features.index], reg, ["Bear", "Stagnant", "Bull"]), width="stretch")
st.dataframe(summ.reset_index().rename(columns={"PctSample": "% of sample", "MeanWTI": "Mean WTI (USD/bbl)",
                                               "MeanMonthlyReturn": "Mean monthly return (%)", "MeanAnnVol": "Mean annualised vol (%)"}),
             width="stretch", hide_index=True)

ui.section("The three regimes, illustrated from the data")
spells = compute_spells(reg)
cols = st.columns(3)
for col, name, tone in zip(cols, ["Bull", "Bear", "Stagnant"], ["bull", "bear", "stag"]):
    with col:
        st.markdown(f"#### {ui.regime_chip(name)}", unsafe_allow_html=True)
        if name in summ.index:
            r = summ.loc[name]
            st.metric("Months", f"{int(r['Months'])} ({r['PctSample']}%)")
            st.metric("Mean WTI", f"${r['MeanWTI']:.2f}")
            st.metric("Mean annualised volatility", f"{r['MeanAnnVol']:.1f}%")
        sp = spells[(spells["Regime"] == name) & (spells["Months"] >= 3)].sort_values("Months", ascending=False).head(4)
        if len(sp):
            st.caption("Longest episodes: " + "; ".join(f"{a:%b %Y}–{b:%b %Y} ({m} mo)" for a, b, m in zip(sp["Start"], sp["End"], sp["Months"])))
        else:
            st.caption("No episode of 3 or more consecutive months.")

ui.dual_explanation(
    "Think of it as two questions asked each month: *has oil risen or fallen a lot over the past year?* and *has it been jumping around a lot?* "
    "Strong, steady gains → Bull. Big falls with wild swings → Bear. Everything else → Stagnant.",
    "**Definitions.** r₁₂ = P_t / P_{t−12} − 1 (trailing 12-month simple return); σ₁₂ = std of the last 12 monthly log-returns × √12.\n\n"
    "Bull: r₁₂ > +10% ∧ σ₁₂ < 40%. Bear: r₁₂ < −10% ∧ σ₁₂ > 30%. Stagnant: otherwise (the two gates cannot both hold, so there is no precedence issue).\n\n"
    "**Properties.** Backward-looking and rule-bound: it needs 12 months of history, cannot flag a turn until a year of data has moved, and is sensitive to the chosen thresholds. "
    "Unlike the HMM it has no probabilistic interpretation and no notion of persistence. April 2020's extreme prints are retained (no winsorisation).")
ui.disclaimer()
