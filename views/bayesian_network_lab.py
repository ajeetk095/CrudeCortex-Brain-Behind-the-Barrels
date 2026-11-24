"""Bayesian Network Lab — discretisation, Hill Climb structure learning, DAG, CPDs and checks."""
import numpy as np
import pandas as pd
import streamlit as st

from core import bn_engine as B
from core import charts as G
from core import config as C
from core import controls, exports, state, ui_components as ui

run = state.get_run()
ds = state.load_dataset(run)
rh = state.load_hmm(ds, run)

ui.page_header("Bayesian Network Lab", "Discretise, learn a dependency structure with Hill Climb search, and read the conditional probability tables.",
               "Bayesian research")
ui.config_banner(run)
with st.expander("Bayesian-network configuration", expanded=not run.bn == C.BNConfig()):
    controls.bn_controls()

panel, bn = state.load_bn(ds, rh, run)
score_name = run.bn.score.upper()
tabs = st.tabs(["Method", "Discretisation", "Structure (DAG)", "Search path", "CPDs", "Checks", "Variants & downloads"])

# ------------------------------------------------------------------ Method
with tabs[0]:
    ui.dual_explanation(
        "A Bayesian Network is a map of which indicators seem to go together with the oil regime. Arrows connect indicators whose behaviour is statistically linked; "
        "tables then say how likely each regime is given the state of its parents. Arrows show learned association, not proof that one thing causes another.",
        f"**Hill Climb search** starts from an empty graph and repeatedly applies the single best add / delete / reverse edge operation that keeps the graph acyclic and respects the maximum of "
        f"{run.bn.max_parents} parents per node, until no operation improves the **{score_name}** score.\n\n"
        "BIC = log L − (k/2)·ln n, with k = Σ (r_i − 1)·q_i free parameters. The score is decomposable, so each node's family is scored separately. "
        "CPDs are maximum-likelihood relative frequencies; inference is exact enumeration over the joint distribution.")
    ui.callout("warn", "Equivalent orientations",
               "Reversing a single link gives the same score and the same joint distribution, so the data cannot say which way an arrow points. "
               "Arrows into the regime node follow a stated convention (ties are broken in favour of links into WTI_Regime); this is not a causal finding.")
    ui.kpis([("Nodes", str(len(bn.nodes)), ", ".join(bn.nodes), ""),
             ("Edges learned", str(len(bn.edges)), "max parents " + str(run.bn.max_parents), "acc"),
             ("Search steps", str(len(bn.history) - 1), "operations accepted", ""),
             (f"Final {score_name}", f"{bn.history[-1]['total_score']:.2f}", "higher is better", "")])

# ------------------------------------------------------------------ Discretisation
with tabs[1]:
    st.markdown("**Discrete dataset** — each continuous series is converted to Low / Medium / High (or Down / Flat / Up for trends).")
    st.dataframe(panel.tail(12), width="stretch")
    rows = []
    for col, key in [(v, k) for k, v in (C.BN_TREND_NODES if run.bn.variables == "trend" else C.BN_LEVEL_NODES).items() if k in run.bn.indicators]:
        vc = panel[col].value_counts()
        rows.append({"Node": col, **{str(s): int(vc.get(s, 0)) for s in B.category_order(col, panel[col].unique())}})
    st.markdown("**Months per category**")
    st.dataframe(pd.DataFrame(rows).fillna(0), width="stretch", hide_index=True)
    st.markdown("**Regime node**")
    st.dataframe(panel[C.TARGET_NODE].value_counts().rename("Months").to_frame().T, width="stretch")
    ui.callout("note", "Method: " + ("33rd / 67th percentiles" if run.bn.discretisation == "report" else "equal-frequency tertiles"),
               "Cut-offs are computed on the full modelled sample, so category boundaries use information from the whole period (a descriptive, not a real-time, discretisation).")

# ------------------------------------------------------------------ DAG
with tabs[2]:
    st.plotly_chart(G.fig_dag(bn), width="stretch")
    et = B.edge_table(bn)
    if et.empty:
        ui.callout("note", "No dependencies retained", f"With these settings the {score_name} penalty outweighs every possible improvement in fit, so the search kept an empty graph (all variables treated as independent).")
    else:
        st.dataframe(et, width="stretch", hide_index=True)
        st.caption("'Score loss if removed' = how much the family score drops without that parent (larger = stronger evidence).")
    if run.is_research_default:
        got = sorted(bn.edges)
        ref = sorted(C.REPORT_REFERENCE["bn_edges_report"])
        (st.success if got == ref else st.warning)(
            f"Submitted report edge set: {ref}. This run: {got}. " + ("Match." if got == ref else "Different — see the Methodology page."))
    ui.callout("warn", "Correlation is not causation",
               "An edge means the variables are conditionally dependent in this sample. It does not show that VIX or any macro indicator moves oil regimes.")

# ------------------------------------------------------------------ Search path
with tabs[3]:
    st.plotly_chart(G.fig_bic_progress(bn.history, score_name), width="stretch")
    st.dataframe(pd.DataFrame(bn.history), width="stretch", hide_index=True)
    with st.expander("Worked idea: one Hill Climb step"):
        st.markdown("From the current graph, every legal single-edge change is scored (only the affected child's family changes). "
                    "The change with the largest score gain is applied; if the best gain is below 1e-9 the search stops. "
                    "This greedy rule can stop in a local optimum, which is why the result is cross-checked against an independent implementation in the test-suite.")

# ------------------------------------------------------------------ CPDs
with tabs[4]:
    node = st.selectbox("Node", bn.nodes, index=bn.nodes.index(C.TARGET_NODE) if C.TARGET_NODE in bn.nodes else 0, key="cpd_node")
    st.plotly_chart(G.fig_cpd(bn, node), width="stretch")
    cf = B.cpd_frame(bn, node)
    st.dataframe(cf.style.format("{:.3f}", na_rep="n/a (unobserved)"), width="stretch")
    pa = bn.parents_of(node)
    st.caption("Parents: " + (", ".join(pa) if pa else "none") + ". Columns with 'n/a' are parent combinations never seen in the data.")
    if C.TARGET_NODE in panel and "VIX_Level" in panel:
        t14 = B.conditional_regime_table(panel)
        st.markdown("**P(regime | VIX level)** (report Table 14 definition)")
        st.dataframe(t14, width="stretch", hide_index=True)
        st.plotly_chart(G.fig_vix_conditional(t14, rh.state_names), width="stretch")
        if run.is_research_default and run.bn.discretisation == "report":
            ref = C.REPORT_REFERENCE["table14"]
            bad = [(l, s) for l in ref for s in ref[l]
                   if l in set(t14["VIX_Level"]) and abs(float(t14.loc[t14.VIX_Level == l, f"P({s})"].iloc[0]) - ref[l][s]) > 0.0015]
            (st.success if not bad else st.warning)("Table 14 reproduced to 3 decimals." if not bad else f"Differences vs report: {bad}")

# ------------------------------------------------------------------ Checks
with tabs[5]:
    try:
        ho = B.holdout_evaluation(panel, bn.edges)
        ui.kpis([("Train / test", f"{ho['n_train']} / {ho['n_test']}", f"{ho['train_period']} → {ho['test_period']}", ""),
                 ("Log-loss (network)", f"{ho['log_loss_model']:.3f}", f"baseline {ho['log_loss_baseline']:.3f}", "acc"),
                 ("Accuracy (network)", f"{ho['accuracy_model']:.1%}", f"baseline {ho['accuracy_baseline']:.1%}", ""),
                 ("Target parents", ", ".join(ho["target_parents"]) or "none", "Laplace α = 1", "")])
        better = ho["log_loss_model"] < ho["log_loss_baseline"]
        ui.callout("note", "How to read this",
                   f"On the last 20% of months the network's regime probabilities {'scored better' if better else 'did not score better'} than the training-sample marginal frequencies. "
                   "This is a chronological descriptive check with the structure learned on the full sample; it is not out-of-sample forecasting evidence.")
    except ValueError as e:
        st.info(str(e))
    ui.callout("plain", "In plain language", "We hide the most recent fifth of the history, learn the tables from the rest, and see whether they describe the hidden fifth better than just using overall frequencies.")

# ------------------------------------------------------------------ Variants
with tabs[6]:
    st.markdown("**Same data, different modelling choices** (computed live; applied settings stay unchanged)")
    rows = []
    for var in ("trend", "level"):
        for disc in ("report", "tertile"):
            p = B.build_discrete_panel(ds.features, rh.labels, ds.master, disc, var, run.bn.indicators)
            m = B.build_bn(p, run.bn.max_parents, run.bn.score)
            rows.append({"Variables": var, "Discretisation": disc, "Edges": len(m.edges),
                         "Edge list": "; ".join(f"{a}→{b}" for a, b in sorted(m.edges)) or "none"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Differences between rows show how sensitive the learned structure is to analyst choices — an important limitation.")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Edges (CSV)", exports.bn_edges_csv(bn), "bn_edges.csv", "text/csv")
    d2.download_button("Search history (CSV)", exports.bn_history_csv(bn), "bn_search_history.csv", "text/csv")
    d3.download_button("CPDs (CSV)", exports.cpds_csv(bn), "bn_cpds.csv", "text/csv")
ui.disclaimer()
