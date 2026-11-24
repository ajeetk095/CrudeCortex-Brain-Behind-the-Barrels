"""Regime Detection Pipeline — Steps 6–7: transformation, parameter learning, Viterbi decoding, latent meaning."""
import numpy as np
import pandas as pd
import streamlit as st

from core import charts as G
from core import config as C
from core import controls, exports, state, ui_components as ui
from core.hmm_engine import regime_agreement, rule_based_regimes, verify_gaussian_hmm

run = state.get_run()
ds = state.load_dataset(run)

ui.page_header("Regime Detection Pipeline", "From raw prices to decoded regimes: transformation, Baum–Welch learning, Viterbi decoding and the meaning of each hidden state.",
               "HMM research · Steps 6–7")
ui.config_banner(run)
with st.expander("HMM configuration — states, restarts, iterations, tolerance, covariance", expanded=not run.hmm == C.HMMConfig()):
    controls.hmm_controls()

rh = state.load_hmm(ds, run)
names = rh.state_names
cmap = G.color_map(names)
alt = not run.is_research_default
if not rh.converged:
    st.warning("The selected model reports non-convergence.")

tabs = st.tabs(["6A Transformation", "6B Parameters", "6C Decoding", "7 Latent meaning", "Persistence", "Convergence", "Verification & export"])

# ---------------------------------------------------------------- 6A
with tabs[0]:
    a = ds.adf
    ui.kpis([("ADF statistic", f"{a['stat']:.4f}", "report: −9.7807" if ds.is_report_window else "alternative window", ""),
             ("p-value", f"{a['p']:.4f}", "stationary at 1%" if a["p"] < 0.01 else "check stationarity", ""),
             ("1% critical value", f"{a['crit_1pct']:.3f}", f"lags {a['lags']}, n = {a['nobs']}", ""),
             ("Feature matrix", f"{ds.features.shape[0]} × 2", ds.fmt_range("features"), "acc")])
    ui.dual_explanation(
        "Raw oil prices trend and drift, which confuses statistical models. We convert each month into two stable numbers: how much the price changed, and how bumpy the last year has been. "
        "Both are then rescaled so that neither dominates just because of its units.",
        "1. **Log-return** r_t = ln(P_t/P_{t−1}) — removes the price trend; the Augmented Dickey–Fuller test rejects a unit root.\n"
        "2. **Volatility** σ¹²_t = √12 · std(r_{t−11..t}) — annualised 12-month rolling standard deviation.\n"
        "3. **Standardisation** z = (x − mean)/std over the modelled sample (StandardScaler) so the Gaussian likelihood is not dominated by scale.\n\n"
        f"The first {ds.window['warmup_rows']} panel months are consumed by this transformation (see Data & Indicators).")
    c1, c2 = st.columns(2)
    c1.plotly_chart(G.fig_returns(ds.features), width="stretch")
    c2.plotly_chart(G.fig_vol(ds.features), width="stretch")

# ---------------------------------------------------------------- 6B
with tabs[1]:
    st.markdown(ui.badge("Model-estimated quantities", "info"), unsafe_allow_html=True)
    ui.kpis([("Log-likelihood", f"{rh.ll:.4f}", "report: −305.9770" if run.is_research_default else "alternative run", "acc"),
             ("EM iterations", str(rh.n_iter_used), "converged" if rh.converged else "not converged", ""),
             ("Parameters", str(rh.n_params), f"AIC {rh.aic:.1f} · BIC {rh.bic:.1f}", ""),
             ("Selected seed", str(int(rh.restarts.loc[rh.restarts['Selected'], 'Seed'].iloc[0])), f"of {rh.config.n_restarts} restarts", "")])
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Initial-state probabilities π**")
        st.dataframe(pd.DataFrame([rh.startprob], columns=names, index=["π"]).round(4), width="stretch")
        st.markdown("**State means (original units)**")
        mt = pd.DataFrame({"Mean monthly log-return (%)": rh.means_orig[:, 0] * 100,
                           "Mean annualised volatility (%)": rh.means_orig[:, 1] * 100}, index=names).round(3)
        mt["Raw hmmlearn index"] = [rh.raw_index_map[n] for n in names]
        st.dataframe(mt, width="stretch")
    with c2:
        st.plotly_chart(G.fig_transition_heatmap(rh.transmat, names), width="stretch")
    st.plotly_chart(G.fig_state_means(rh), width="stretch")
    st.plotly_chart(G.fig_state_covariances(rh), width="stretch")
    st.plotly_chart(G.fig_feature_space(ds.features, rh), width="stretch")
    with st.expander("Covariance matrices — numbers (original units: return², return·vol, vol²)"):
        for i, n in enumerate(names):
            st.markdown(f"**{n}**")
            st.dataframe(pd.DataFrame(rh.covars_orig[i], index=["Return", "Volatility"], columns=["Return", "Volatility"]).style.format("{:.5f}"), width="stretch")
    ui.dual_explanation(
        "The model learned, by itself, what a 'typical' month looks like in each regime, how likely the market is to stay in a regime or switch, and how spread out months are around that typical value. "
        "The transition matrix's diagonal shows stickiness: values near 1 mean the regime usually lasts.",
        "**Baum–Welch (EM).** E-step: forward–backward gives γ_t(j) and ξ_t(i,j). M-step: π̂, â_ij = Σξ/Σγ, μ̂_j = Σγ x/Σγ, Σ̂_j = Σγ(x−μ̂)(x−μ̂)ᵀ/Σγ. "
        "Iterations stop when the log-likelihood improves by less than the tolerance or at the iteration limit. Of the restarts, the best **converged** log-likelihood is kept. "
        "Means are reported in standardised units internally and converted back to original units here.")

# ---------------------------------------------------------------- 6C
with tabs[2]:
    st.markdown(ui.badge("Model-estimated state sequence", "info"), unsafe_allow_html=True)
    st.plotly_chart(G.fig_regime_timeline(rh.labels, names, "Viterbi-decoded regime timeline"), width="stretch")
    st.plotly_chart(G.fig_price(ds.master.loc[ds.features.index], rh.labels, names), width="stretch")
    st.plotly_chart(G.fig_viterbi_posterior(rh), width="stretch")
    agree = float((rh.posteriors.idxmax(axis=1) == rh.labels).mean())
    st.caption(f"The most probable state per month (posterior argmax) equals the Viterbi state in {agree:.1%} of months. "
               "Viterbi decodes the best whole path; posteriors give per-month uncertainty.")
    ui.dual_explanation(
        "For every month the model picks the regime that, taken together with the other months, best explains the whole history. "
        "The lower chart shows how confident it was: months where one colour fills the whole height are clear-cut, mixed months are ambiguous.",
        "Viterbi: δ_t(j) = max_i[δ_{t−1}(i)a_ij]·b_j(x_t), ψ_t(j) = argmax_i(…), q*_T = argmax δ_T, q*_t = ψ_{t+1}(q*_{t+1}). "
        "Decoding uses all months (smoothing), so a month's label can depend on later data; it is **not** a real-time signal.")

# ---------------------------------------------------------------- 7
with tabs[3]:
    st.markdown(ui.badge("Observed data grouped by decoded regime", "ok"), unsafe_allow_html=True)
    stt = rh.state_stats.rename(columns={"MeanReturn": "Mean log-return (%)", "MeanVol": "Mean ann. vol (%)",
                                         "MeanVIX": "Mean VIX", "PctSample": "% of sample"}).round(2)
    st.dataframe(stt, width="stretch")
    if run.is_research_default:
        ref = C.REPORT_REFERENCE
        chk = pd.DataFrame({
            "Report months": pd.Series(ref["hmm_regime_counts"]), "This run": rh.labels.value_counts(),
            "Report mean VIX": pd.Series(ref["hmm_mean_vix"]), "This run VIX": rh.state_stats["MeanVIX"].round(1)}).reindex(names)
        st.markdown("**Comparison with the submitted report (Tables 13/14)**")
        st.dataframe(chk, width="stretch")
    hv = rh.state_stats["MeanVol"].idxmax()
    ui.callout("warn", "Check the names against the behaviour",
               f"Names follow the ordering of mean returns only. In this run the highest-volatility state is **{hv}** "
               f"({rh.state_stats.loc[hv, 'MeanVol']:.0f}% annualised, mean VIX {rh.state_stats.loc[hv, 'MeanVIX']:.1f}). "
               "A label such as 'Stagnant' should therefore be read as 'the middle-return state', not necessarily 'quiet'.")
    cols = st.columns(len(names))
    for col, n in zip(cols, names):
        r = rh.state_stats.loc[n]
        col.markdown(f'<div class="cc-card"><h4>{ui.regime_chip(n, cmap)}</h4><p>{int(r["Months"])} months · mean return {r["MeanReturn"]:+.2f}%/month · '
                     f'volatility {r["MeanVol"]:.1f}% · VIX {r["MeanVIX"]:.1f}</p></div>', unsafe_allow_html=True)
    st.plotly_chart(G.fig_regime_distribution({"HMM": rh.labels, "Rule-based benchmark": ds.master.pipe(rule_based_regimes).loc[ds.features.index]}
                                              if names == C.REGIME_ORDER_3 else {"HMM": rh.labels}, names), width="stretch")
    if names == C.REGIME_ORDER_3:
        rule = rule_based_regimes(ds.master).loc[ds.features.index]
        xt = regime_agreement(rh.labels, rule)
        st.markdown("**HMM regimes (rows) vs rule-based benchmark (columns), months**")
        st.dataframe(xt, width="stretch")
        st.caption("The two classifications are different tools and are not expected to coincide.")
    ui.callout("plain", "In plain language",
               "Having labelled the months, we check what each group actually looked like. This is the honest test of whether the model's 'moods' mean anything: do they differ in returns, volatility and market fear?")

# ---------------------------------------------------------------- persistence
with tabs[4]:
    st.dataframe(rh.duration_summary.round(2), width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(G.fig_duration(rh.spells, names), width="stretch")
    with c2:
        st.markdown("**All spells**")
        sp = rh.spells.copy()
        sp["Start"], sp["End"] = sp["Start"].dt.strftime("%Y-%m"), sp["End"].dt.strftime("%Y-%m")
        st.dataframe(sp, width="stretch", hide_index=True, height=300)
    ui.dual_explanation(
        "A regime that lasts many months in a row is 'persistent'. The model's own expectation of how long a regime lasts can be compared with how long the observed spells actually were.",
        "Under a first-order Markov chain the dwell time in state i is geometric with mean 1/(1 − a_ii). The table compares that with the mean of observed spells; large gaps with few spells indicate that the sample is too short to pin down persistence precisely.")

# ---------------------------------------------------------------- convergence
with tabs[5]:
    st.plotly_chart(G.fig_convergence(rh), width="stretch")
    st.dataframe(rh.restarts.round(4), width="stretch", hide_index=True)
    r = rh.restarts.dropna(subset=["Log-likelihood"])
    best = r["Log-likelihood"].max()
    near = int((r["Log-likelihood"] > best - 0.01).sum())
    ui.callout("note", "Local optima are real",
               f"{near} of {len(r)} restarts reached the best log-likelihood (within 0.01); the others stopped at lower values "
               f"(worst {r['Log-likelihood'].min():.2f} vs best {best:.2f}). EM only guarantees a local optimum, which is why several random restarts are used.")

# ---------------------------------------------------------------- verification
with tabs[6]:
    v = verify_gaussian_hmm(rh, ds.X_std)
    st.markdown("**Independent check** — the log-space Forward, Backward and Viterbi routines written for this project are re-run on the fitted parameters and compared with hmmlearn.")
    ui.kpis([("Log-likelihood gap", f"{v['loglik_abs_gap']:.1e}", "own forward vs hmmlearn", ""),
             ("Forward = Backward", f"{v['forward_backward_gap']:.1e}", "absolute gap", ""),
             ("Viterbi agreement", f"{v['viterbi_agreement']:.1%}", "own vs hmmlearn", ""),
             ("Posterior max gap", f"{v['posterior_max_gap']:.1e}", "own vs hmmlearn", "")])
    st.markdown("**Downloads**")
    rule = rule_based_regimes(ds.master).loc[ds.features.index]
    d1, d2, d3 = st.columns(3)
    d1.download_button("Regime labels + posteriors (CSV)", exports.regimes_csv(ds, rh, rule), "regime_labels.csv", "text/csv")
    d2.download_button("Transition matrix (CSV)", exports.transition_csv(rh), "hmm_transition_matrix.csv", "text/csv")
    d3.download_button("State statistics (CSV)", exports.state_stats_csv(rh), "hmm_state_statistics.csv", "text/csv")
ui.disclaimer()
