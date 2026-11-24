"""Methodology & Limitations — what the report did, what this app adds, discrepancy register, limitations."""
import pandas as pd
import streamlit as st

from core import config as C
from core import state, ui_components as ui
from core.data_gwp2 import vix_method_comparison

run = state.get_run()
ds = state.load_dataset(run)
rh = state.load_hmm(ds, run)
panel, bn = state.load_bn(ds, rh, run)

ui.page_header("Methodology & Limitations", "What was preserved from the submitted research, what was added, which discrepancies were resolved and where the method is weak.", "Results & documentation")
ui.config_banner(run)

ui.section("Original methodology (preserved)")
st.markdown(
    "1. Four toy-HMM algorithms (Forward, Backward, Viterbi, Baum–Welch) on a small worked example.\n"
    "2. Rule-based Bull / Bear / Stagnant benchmark from return and volatility thresholds.\n"
    "3. Three-state Gaussian HMM, full covariance, 200 EM iterations, tolerance 1e-4, 10 random restarts, best converged log-likelihood.\n"
    "4. Five-node Bayesian Network learned with Hill Climb search and BIC on quantile-discretised data; CPDs and exact inference.")
ui.section("Enhancements in this application")
st.markdown(
    "- Data pipeline with timeouts, retries, cache, validation and visible provenance; offline snapshot as the default source.\n"
    "- Log-space (numerically stable) Forward/Backward/Viterbi, independently checked against hmmlearn.\n"
    "- Corrected Hill Climb parent handling and score definitions; independently cross-checked against pgmpy (development-time only).\n"
    "- Interactive controls with explicit Run / Reset, all deviations from the report flagged.\n"
    "- CSV/ZIP exports and a reproducible configuration summary.")

ui.section("Discrepancy register")
reg = pd.DataFrame([
    ("D1", "Date window", "Report text mentions 2005/2006 starts; the trade-weighted dollar index (DTWEXBGS) begins 2006-01, so the common panel starts 2006-01 and the first model month is 2007-01 after a 12-month warm-up.", "Window derived from data and displayed; user can change the start."),
    ("D2", "VIX source", "Notebook used FRED VIXCLS; a Yahoo ^VIX series can differ slightly.", "FRED is primary; Yahoo is last-resort fallback and flagged."),
    ("D3", "VIX aggregation", "Monthly VIX can be month-end close or monthly mean.", "Default month-end close (reproduces report); alternative selectable and compared below."),
    ("D4", "Discretisation", "Report cut-offs at 33rd/67th percentile vs equal-frequency qcut.", "Both offered; default = report."),
    ("D5", "BN variables", "Report text implies levels; executed notebook used VIX level + macro trends.", "Default = notebook (trend); level variant selectable."),
    ("D6", "Edge orientation", "Opposite orientations of one link are score-equivalent.", "Convention: links into WTI_Regime; stated on the BN page."),
    ("D7", "Snapshot provenance", "The bundled FRED CSVs came with the project; their retrieval date is not recorded.", "Manifest with checksums; refresh script provided; live retrieval available."),
], columns=["ID", "Topic", "Issue", "Resolution"])
st.dataframe(reg, width="stretch", hide_index=True)

cmp_ = vix_method_comparison(ds)
if cmp_ is not None:
    with st.expander("Evidence for D3: month-end close vs monthly mean VIX"):
        st.dataframe(cmp_.describe().round(2), width="stretch")
        st.caption(f"Mean absolute difference: {cmp_['Difference (close - mean)'].abs().mean():.2f} VIX points; maximum: {cmp_['Difference (close - mean)'].abs().max():.2f}.")

ui.section("Limitations")
r = rh.restarts.dropna(subset=["Log-likelihood"])
n = len(rh.labels)
lim = [
    ("Small sample", f"{n} monthly observations for {rh.n_params} HMM parameters ({n / rh.n_params:.1f} per parameter); regimes with few months have uncertain parameters."),
    ("Local optima", f"EM restarts ended between {r['Log-likelihood'].min():.2f} and {r['Log-likelihood'].max():.2f} log-likelihood."),
    ("Gaussian emissions", "Oil returns are fat-tailed; a Gaussian per regime is an approximation."),
    ("Hindsight labels", "Viterbi/posterior decoding uses the whole sample; regimes are not real-time signals."),
    ("Label semantics", "Names follow mean-return order only; check the volatility of each state (Pipeline, step 7)."),
    ("Stationarity & homogeneity", "Transition probabilities are assumed constant over 18+ years that include several structural breaks."),
    ("Learned structure on same sample", "Regimes and the BN structure come from the same data; no true out-of-sample test."),
    ("Discretisation loss", "Collapsing series to three categories discards information; results depend on cut-offs."),
    ("Greedy search", "Hill Climb can stop in a local optimum and is sensitive to the starting graph."),
    ("Score equivalence", "Edge directions are not identified by the data."),
    ("Association ≠ causation", "Edges and CPDs are conditional associations."),
    ("Data provenance", "Snapshot is a historical copy; FRED series are revised over time; live retrieval could not be exercised in the build sandbox."),
    ("Monthly frequency", "Intramonth dynamics are invisible; month-end vs mean VIX changes results."),
    ("Scope", "Single commodity (WTI), five indicators, no transaction, forecasting or portfolio component — by design."),
    ("No investment advice", "Nothing here is a trading signal or recommendation."),
]
st.dataframe(pd.DataFrame(lim, columns=["Limitation", "Evidence / comment"]), width="stretch", hide_index=True)
ui.disclaimer()
