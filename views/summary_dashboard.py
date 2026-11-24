"""Summary Dashboard — headline results, report comparison, audience interpretations, ZIP export."""
import pandas as pd
import streamlit as st

from core import charts as G
from core import config as C
from core import exports, state, ui_components as ui

run = state.get_run()
an = state.load_analysis(run)
ds, rh, bn, rule, panel = an.ds, an.rh, an.bn, an.rule, an.panel
names = rh.state_names
cmap = G.color_map(names)

ui.page_header("Summary Dashboard", "The whole analysis on one page, generated from the currently applied configuration.", "Results & documentation")
ui.config_banner(run)

counts = rh.labels.value_counts().reindex(names).fillna(0).astype(int)
ui.kpis([("Modelled months", str(len(rh.labels)), ds.fmt_range("features"), ""),
         ("HMM log-likelihood", f"{rh.ll:.3f}", f"{rh.n_iter_used} EM iterations", "acc"),
         ("Regimes", " / ".join(f"{n[0]}{counts[n]}" for n in names), "months per regime", ""),
         ("BN edges", str(len(bn.edges)), "; ".join(f"{a}→{b}" for a, b in sorted(bn.edges)) or "none", "")])

c1, c2 = st.columns([3, 2])
c1.plotly_chart(G.fig_regime_timeline(rh.labels, names, "Decoded regimes"), width="stretch")
c2.plotly_chart(G.fig_transition_heatmap(rh.transmat, names), width="stretch")
c3, c4 = st.columns(2)
c3.plotly_chart(G.fig_dag(bn, height=380), width="stretch")
if "VIX_Level" in panel:
    c4.plotly_chart(G.fig_vix_conditional(__import__("core.bn_engine", fromlist=["x"]).conditional_regime_table(panel), names), width="stretch")

ui.section("Comparison with the submitted report")
if run.is_research_default:
    ref = C.REPORT_REFERENCE
    cmp_ = pd.DataFrame([
        ("Log-likelihood", f"{ref['hmm_loglik']:.4f}", f"{rh.ll:.4f}", abs(rh.ll - ref["hmm_loglik"]) < 5e-3),
        ("Regime counts", str(ref["hmm_regime_counts"]), str(counts.to_dict()), counts.to_dict() == ref["hmm_regime_counts"]),
        ("ADF statistic", f"{ref['adf_statistic']:.4f}", f"{ds.adf['stat']:.4f}", abs(ds.adf["stat"] - ref["adf_statistic"]) < 5e-4),
        ("Feature matrix", str(ref["feature_shape"]), str(ds.features.shape), tuple(ds.features.shape) == ref["feature_shape"]),
        ("BN edges", str(ref["bn_edges_report"]), str(sorted(bn.edges)), sorted(bn.edges) == sorted(ref["bn_edges_report"])),
    ], columns=["Quantity", "Report", "This run", "Match"])
    cmp_["Match"] = cmp_["Match"].map({True: "✓", False: "✗"})
    st.dataframe(cmp_, width="stretch", hide_index=True)
    st.caption("Values are recomputed from data; the report column is for comparison only. A ✗ indicates the data snapshot or a library version differs.")
else:
    ui.callout("warn", "Alternative run", "Report comparison is shown only for the research-default configuration. Deviations: " + "; ".join(run.deviations()))

ui.section("Interpretation by audience")
hv = rh.state_stats["MeanVol"].idxmax()
sticky = max(names, key=lambda n: rh.transmat[names.index(n), names.index(n)])
si = names.index(sticky)
facts = {
    "persistence": f"The most persistent regime is {sticky}: the model gives a {rh.transmat[si, si]:.0%} chance of staying in it next month (mean observed spell {rh.duration_summary.loc[sticky, 'Mean spell (months)']:.1f} months).",
    "transition": f"Self-transition probabilities: " + ", ".join(f"{n} {rh.transmat[i, i]:.0%}" for i, n in enumerate(names)) + ".",
    "energy": f"The {hv} regime has the highest volatility ({rh.state_stats.loc[hv, 'MeanVol']:.0f}% annualised) and covers {counts[hv]} of {counts.sum()} months.",
    "quant": f"Best of {rh.config.n_restarts} restarts; log-likelihood {rh.ll:.3f}, AIC {rh.aic:.1f}, BIC {rh.bic:.1f}; {rh.n_params} free parameters for {len(rh.labels)} observations.",
    "vix": "Mean VIX by regime: " + ", ".join(f"{n} {rh.state_stats.loc[n, 'MeanVIX']:.1f}" for n in names) + ".",
}
ui.audience_panels(facts)

ui.section("Export")
st.download_button("Download everything (ZIP: CSVs, configuration JSON, summary)", exports.build_zip(run, ds, rh, rule, bn),
                   "crudecortex_gwp2_results.zip", "application/zip", type="primary")
with st.expander("Summary text (Markdown)"):
    st.markdown(exports.summary_markdown(run, ds, rh, bn))
ui.disclaimer()
