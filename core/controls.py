"""Interactive research controls (Data, HMM, Bayesian network).

Widgets edit *pending* values; the applied configuration only changes when the
visitor presses **Run analysis**, so expensive models are never refitted on a
stray click. Every panel offers **Reset to research defaults**.
"""
from __future__ import annotations

import dataclasses

import streamlit as st

from . import config as C
from . import state
from .config import BNConfig, DataConfig, HMMConfig
from .data_gwp2 import Dataset

TOL_OPTIONS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def data_controls(ds: Dataset) -> None:
    run = state.get_run()
    d = run.data
    months = [f"{m:%Y-%m}" for m in ds.full_master.index[C.WARMUP_MONTHS:]]
    st.caption(f"Selectable modelled months: **{months[0]} → {months[-1]}**. The first {C.WARMUP_MONTHS} panel months "
               "are consumed by the return and rolling-volatility warm-up, so the earliest modellable month is "
               f"{months[0]}. Frequency is fixed at monthly: the HMM features are monthly by construction.")
    cur = (d.feature_start[:7] if d.feature_start[:7] in months else months[0],
           d.feature_end[:7] if d.feature_end[:7] in months else months[-1])
    rng = st.select_slider("Modelled period (first and last month)", options=months, value=cur, key="w_data_rng")
    c1, c2 = st.columns(2)
    agg = c1.radio("VIX monthly aggregation", ["month_end_close", "monthly_mean"],
                   index=["month_end_close", "monthly_mean"].index(d.vix_aggregation), key="w_data_agg",
                   format_func=lambda v: "Month-end close (report method)" if v == "month_end_close" else "Monthly mean (earlier app variant)",
                   help="The report used Yahoo monthly closes. FRED's daily VIXCLS resampled to the last trading day reproduces those values.")
    mode = c2.radio("Data source", ["snapshot", "live"], index=["snapshot", "live"].index(d.data_mode), key="w_data_mode",
                    format_func=lambda v: "Bundled snapshot (reproducible)" if v == "snapshot" else "Live FRED (refreshable)",
                    help="Live mode downloads the latest public FRED CSVs. History can be revised, so results may drift from the report.")
    b1, b2, b3 = st.columns(3)
    if b1.button("Run analysis", type="primary", key="btn_run_data"):
        new = dataclasses.replace(d, feature_start=f"{rng[0]}-01", feature_end=f"{rng[1]}-01",
                                  vix_aggregation=agg, data_mode=mode)
        state.set_run(dataclasses.replace(run, data=new))
        st.rerun()
    if b2.button("Refresh data from FRED", key="btn_refresh"):
        state.request_refresh()
        st.rerun()
    if b3.button("Reset to research defaults", key="btn_reset_data"):
        state.reset_run("data")
        for k in ("w_data_rng", "w_data_agg", "w_data_mode"):
            st.session_state.pop(k, None)
        st.rerun()


def hmm_controls() -> None:
    run = state.get_run()
    h = run.hmm
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Hidden states", 2, 5, h.n_states, key="w_hmm_n",
                  help="Research default: 3 (Bull / Bear / Stagnant). Other counts get generic state names.")
    r = c2.slider("Random restarts", 1, 30, h.n_restarts, key="w_hmm_r")
    it = c3.slider("Max EM iterations", 20, 500, h.n_iter, step=10, key="w_hmm_it")
    c4, c5 = st.columns(2)
    tol = c4.select_slider("Convergence tolerance", options=TOL_OPTIONS, value=h.tol if h.tol in TOL_OPTIONS else 1e-4,
                           format_func=lambda v: f"{v:.0e}", key="w_hmm_tol")
    cov = c5.selectbox("Covariance structure", ["full", "diag", "tied", "spherical"],
                       index=["full", "diag", "tied", "spherical"].index(h.covariance_type), key="w_hmm_cov",
                       help="Research default: full covariance (return and volatility may be correlated within a state).")
    if n != 3:
        st.info("Experimental: with a state count other than 3 the Bull/Bear/Stagnant naming does not apply; "
                "states are ordered by mean return (State 1 = lowest).")
    b1, b2 = st.columns(2)
    if b1.button("Run analysis", type="primary", key="btn_run_hmm"):
        state.set_run(dataclasses.replace(run, hmm=HMMConfig(n, r, it, float(tol), cov)))
        st.rerun()
    if b2.button("Reset to research defaults", key="btn_reset_hmm"):
        state.reset_run("hmm")
        for k in ("w_hmm_n", "w_hmm_r", "w_hmm_it", "w_hmm_tol", "w_hmm_cov"):
            st.session_state.pop(k, None)
        st.rerun()


def bn_controls() -> None:
    run = state.get_run()
    b = run.bn
    c1, c2 = st.columns(2)
    mp = c1.slider("Maximum parents per node", 1, 3, b.max_parents, key="w_bn_mp")
    score = c2.selectbox("Search score", ["bic", "aic"], index=["bic", "aic"].index(b.score), key="w_bn_score",
                         format_func=str.upper, help="Research default: BIC (stronger complexity penalty).")
    c3, c4 = st.columns(2)
    var = c3.radio("Variable definition", ["trend", "level"], index=["trend", "level"].index(b.variables), key="w_bn_var",
                   format_func=lambda v: "Report / notebook: VIX level + macro trends" if v == "trend" else "Application variant: all macro levels",
                   help="Levels of IndPro, CPI and DXY share long-run trends, which can create dependence that is not an economic link.")
    disc = c4.radio("Discretisation", ["report", "tertile"], index=["report", "tertile"].index(b.discretisation), key="w_bn_disc",
                    format_func=lambda v: "33rd / 67th percentile cut-offs (report)" if v == "report" else "Equal-frequency tertiles (qcut)")
    ind = st.multiselect("Macro indicators in the network", list(C.MACRO_KEYS), default=list(b.indicators), key="w_bn_ind",
                         format_func=lambda k: C.SERIES[k].label)
    if not ind:
        st.warning("Select at least one macro indicator.")
    b1, b2 = st.columns(2)
    if b1.button("Run analysis", type="primary", key="btn_run_bn", disabled=not ind):
        state.set_run(dataclasses.replace(run, bn=BNConfig(mp, score, disc, var, tuple(k for k in C.MACRO_KEYS if k in ind))))
        st.rerun()
    if b2.button("Reset to research defaults", key="btn_reset_bn"):
        state.reset_run("bn")
        for k in ("w_bn_mp", "w_bn_score", "w_bn_var", "w_bn_disc", "w_bn_ind"):
            st.session_state.pop(k, None)
        st.rerun()
