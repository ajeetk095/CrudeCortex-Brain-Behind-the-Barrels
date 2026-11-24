"""Conditional Probability Explorer — exact inference in the learned network."""
import pandas as pd
import streamlit as st

from core import bn_engine as B
from core import charts as G
from core import config as C
from core import exports, state, ui_components as ui

run = state.get_run()
ds = state.load_dataset(run)
rh = state.load_hmm(ds, run)
panel, bn = state.load_bn(ds, rh, run)

ui.page_header("Conditional Probability Explorer", "Choose evidence, read the probabilities. Exact network inference next to plain relative frequencies.", "Bayesian research")
ui.config_banner(run)
ui.dual_explanation(
    "Pick what you know about the market (for example, VIX is High) and see how the regime frequencies change. It answers 'in this history, how were regimes distributed when …?', not 'what will happen next?'.",
    "P(target | evidence) is computed by enumerating the factorised joint P = ΠP(X_i | Pa_i) and normalising. The **empirical** column uses only the months that match all evidence exactly; "
    "the two agree when the network is saturated for the question and differ when the network's independence assumptions matter.")

target = st.selectbox("Target variable", bn.nodes, index=bn.nodes.index(C.TARGET_NODE) if C.TARGET_NODE in bn.nodes else 0, key="ce_target")
others = [n for n in bn.nodes if n != target]
chosen = st.multiselect("Evidence variables", others, default=[n for n in others if n == "VIX_Level"][:1], key="ce_ev")
evidence = {}
cols = st.columns(max(1, len(chosen)))
for c, n in zip(cols, chosen):
    evidence[n] = c.selectbox(n, bn.states[n], key=f"ce_val_{n}")

try:
    net = B.infer_network(bn, target, evidence)
    emp, n_match = B.infer_empirical(bn, target, evidence)
except B.InferenceError as e:
    st.error(f"Inference could not be run: {e}")
    st.stop()

ui.kpis([("Matching months", str(n_match), "rows meeting all evidence", "acc" if n_match >= 10 else ""),
         ("Evidence", ", ".join(f"{k}={v}" for k, v in evidence.items()) or "none", "", ""),
         ("Most likely (network)", str(net.idxmax()), f"{net.max():.1%}", "")])
if n_match == 0:
    st.warning("No month in the sample matches this evidence combination; the empirical column is empty and the network answer rests on its factorisation alone.")
elif n_match < 10:
    st.warning(f"Only {n_match} matching months — treat the empirical frequencies as very uncertain.")

st.plotly_chart(G.fig_conditional_stack(net, emp, f"P({target} | evidence)"), width="stretch")
tbl = pd.DataFrame({"Network (exact)": net, "Empirical (matching months)": emp}).reindex(net.index)
st.dataframe(tbl.style.format("{:.3f}", na_rep="–"), width="stretch")
st.download_button("Download this query (CSV)", exports.inference_csv(net, emp, evidence, target, n_match), "conditional_query.csv", "text/csv")
ui.callout("warn", "Reading conditional probabilities", "These are descriptive frequencies from past data. They are not forecasts and carry no causal claim.")
ui.disclaimer()
