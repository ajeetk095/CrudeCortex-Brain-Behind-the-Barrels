"""CSV / JSON / Markdown exports and the reproducible configuration summary.

All functions are pure (no Streamlit) and return ``bytes`` or plain Python
objects, so they can be unit-tested and reused from notebooks.
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from typing import Optional

import pandas as pd

from . import config as C
from .bn_engine import BNModel, cpds_long, edge_table
from .config import RunConfig
from .data_gwp2 import Dataset
from .hmm_engine import RegimeHMM

_PKGS = ["streamlit", "pandas", "numpy", "scipy", "scikit-learn", "statsmodels",
         "hmmlearn", "plotly", "requests", "yfinance"]


def to_csv_bytes(df: pd.DataFrame, index: bool = True) -> bytes:
    return df.to_csv(index=index).encode("utf-8")


def package_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for p in _PKGS:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "not installed"
    return out


def master_csv(ds: Dataset) -> bytes:
    return to_csv_bytes(ds.master)


def features_csv(ds: Dataset) -> bytes:
    f = ds.features.copy()
    f[["WTI_ret_z", "WTI_vol12_z"]] = ds.X_std
    return to_csv_bytes(f)


def regimes_csv(ds: Dataset, rh: RegimeHMM, rule: pd.Series) -> bytes:
    out = pd.DataFrame({"HMM_regime_viterbi": rh.labels, "RuleBased_regime": rule.reindex(rh.labels.index)})
    for c in rh.posteriors.columns:
        out[f"P({c}|data)_smoothed"] = rh.posteriors[c]
    return to_csv_bytes(out)


def transition_csv(rh: RegimeHMM) -> bytes:
    return to_csv_bytes(pd.DataFrame(rh.transmat, index=[f"from {n}" for n in rh.state_names],
                                     columns=[f"to {n}" for n in rh.state_names]))


def state_stats_csv(rh: RegimeHMM) -> bytes:
    t = rh.state_stats.copy()
    t["InitialProb"] = rh.startprob
    t["MeanReturn_model_pct"] = rh.means_orig[:, 0] * 100
    t["MeanVol_model_pct"] = rh.means_orig[:, 1] * 100
    t["SelfTransition"] = [rh.transmat[i, i] for i in range(len(rh.state_names))]
    return to_csv_bytes(t.join(rh.duration_summary[["Spells", "Mean spell (months)"]]))


def bn_edges_csv(bn: BNModel) -> bytes:
    return to_csv_bytes(edge_table(bn), index=False)


def bn_history_csv(bn: BNModel) -> bytes:
    return to_csv_bytes(pd.DataFrame(bn.history), index=False)


def cpds_csv(bn: BNModel) -> bytes:
    return to_csv_bytes(cpds_long(bn), index=False)


def inference_csv(dist_network: pd.Series, dist_empirical: Optional[pd.Series], evidence: dict,
                  target: str, n_rows: int = 0) -> bytes:
    df = pd.DataFrame({"network_exact": dist_network})
    if dist_empirical is not None and len(dist_empirical):
        df["empirical_matching_rows"] = dist_empirical
    df.index.name = target
    df["evidence"] = "; ".join(f"{k}={v}" for k, v in evidence.items()) or "(none)"
    df["n_matching_rows"] = n_rows
    return to_csv_bytes(df)


def config_summary(run: RunConfig, ds: Dataset, rh: Optional[RegimeHMM] = None,
                   bn: Optional[BNModel] = None) -> dict:
    """Reproducible description of an analysis run."""
    s: dict = {
        "project": C.PROJECT_NAME, "tagline": C.TAGLINE, "course": f"{C.COURSE} — {C.ASSIGNMENT}, {C.GROUP}",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "configuration_is_research_default": run.is_research_default,
        "deviations_from_report": run.deviations(),
        "configuration": run.as_dict(),
        "data": {
            "master_panel_shape": list(ds.master.shape), "feature_matrix_shape": list(ds.features.shape),
            "panel_range": ds.fmt_range("master"), "model_range": ds.fmt_range("features"),
            "vix_aggregation_effective": ds.vix_aggregation_effective,
            "panel_fingerprint": ds.fingerprint,
            "adf_statistic": ds.adf["stat"], "adf_p_value": ds.adf["p"],
            "series": {r["Series ID"]: {"origin": r["Origin"], "last_retrieval": r["Last retrieval"],
                                         "start": r["Start date"], "end": r["End date"],
                                         "observations": r["Observations"]}
                       for _, r in ds.status.iterrows()},
            "warnings": ds.warnings,
        },
        "software": package_versions(),
        "note": "Descriptive historical analysis; not a forecast or investment recommendation.",
    }
    if rh is not None:
        s["hmm"] = {"log_likelihood": rh.ll, "converged": rh.converged, "em_iterations": rh.n_iter_used,
                    "n_parameters": rh.n_params, "aic": rh.aic, "bic": rh.bic,
                    "regime_months": rh.labels.value_counts().reindex(rh.state_names).fillna(0).astype(int).to_dict(),
                    "selected_seed": int(rh.restarts.loc[rh.restarts["Selected"], "Seed"].iloc[0])}
    if bn is not None:
        s["bayesian_network"] = {"edges": sorted(f"{a} -> {b}" for a, b in bn.edges),
                                 "nodes": bn.nodes, "search_steps": len(bn.history) - 1,
                                 "final_score": bn.history[-1]["total_score"]}
    return s


def summary_markdown(run: RunConfig, ds: Dataset, rh: RegimeHMM, bn: BNModel) -> str:
    """Plain-language analysis summary assembled from the computed results."""
    counts = rh.labels.value_counts().reindex(rh.state_names).fillna(0).astype(int)
    lines = [
        f"# CrudeCortex analysis summary",
        "",
        f"*{C.COURSE} · {C.ASSIGNMENT} · {C.GROUP}* — generated "
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        "",
        "**Configuration:** " + ("research defaults (reproduces the submitted report)."
                                 if run.is_research_default else
                                 "ALTERNATIVE RUN — not the submitted report. " + "; ".join(run.deviations())),
        "",
        f"**Data:** monthly panel {ds.fmt_range('master')} ({ds.master.shape[0]} rows × {ds.master.shape[1]}); "
        f"modelled months {ds.fmt_range('features')} ({ds.features.shape[0]}). "
        f"VIX = {ds.vix_aggregation_effective.replace('_', ' ')}.",
        "",
        f"**Gaussian HMM ({rh.config.n_states} states, {rh.config.covariance_type} covariance):** "
        f"log-likelihood {rh.ll:.4f}, converged={rh.converged} in {rh.n_iter_used} EM iterations.",
        "",
        "| Regime | Months | Share |", "|---|---:|---:|",
    ]
    tot = int(counts.sum())
    for n, c in counts.items():
        lines.append(f"| {n} | {c} | {c / tot:.1%} |")
    lines += ["", "**Bayesian network:** " + (", ".join(f"{a} → {b}" for a, b in sorted(bn.edges)) or "no edges retained"),
              "", "_Descriptive analysis of historical data. Regime labels are model-based classifications of "
              "past months, not forecasts or investment advice. Learned dependencies are conditional "
              "associations, not causal effects._"]
    return "\n".join(lines)


def build_zip(run: RunConfig, ds: Dataset, rh: RegimeHMM, rule: pd.Series, bn: BNModel) -> bytes:
    """All tabular exports plus the configuration summary in one ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("master_panel.csv", master_csv(ds))
        z.writestr("feature_matrix.csv", features_csv(ds))
        z.writestr("regime_labels.csv", regimes_csv(ds, rh, rule))
        z.writestr("hmm_transition_matrix.csv", transition_csv(rh))
        z.writestr("hmm_state_statistics.csv", state_stats_csv(rh))
        z.writestr("bn_edges.csv", bn_edges_csv(bn))
        z.writestr("bn_search_history.csv", bn_history_csv(bn))
        z.writestr("bn_cpds.csv", cpds_csv(bn))
        z.writestr("data_sources.csv", to_csv_bytes(ds.status, index=False))
        z.writestr("configuration_summary.json",
                   json.dumps(config_summary(run, ds, rh, bn), indent=2, default=str))
        z.writestr("analysis_summary.md", summary_markdown(run, ds, rh, bn))
    return buf.getvalue()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
