"""Streamlit glue: applied run configuration (session state) and cached computation.

Design
------
* The **applied** configuration lives in ``st.session_state['cc_run']``. Widgets
  edit *pending* values; nothing is recomputed until the visitor presses
  "Run analysis" (which writes the applied configuration).
* Datasets use ``st.cache_data`` (keyed by the data configuration, TTL 6 h).
* Fitted models use ``st.cache_resource`` keyed by (dataset fingerprint,
  model-configuration JSON), so changing any parameter triggers exactly one
  refit and an unchanged configuration never refits.
* Cached model objects are shared between sessions and must be treated as
  read-only.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st

from . import config as C
from .bn_engine import BNModel, build_bn, build_discrete_panel
from .config import BNConfig, DataConfig, HMMConfig, RunConfig
from .data_gwp2 import Dataset, get_dataset
from .data_sources import DataError
from .hmm_engine import ModelFitError, RegimeHMM, fit_regime_hmm, rule_based_regimes

RUN_KEY = "cc_run"


# --------------------------------------------------------------------------
# Applied configuration
# --------------------------------------------------------------------------

def get_run() -> RunConfig:
    if RUN_KEY not in st.session_state:
        st.session_state[RUN_KEY] = C.DEFAULT_RUN
    return st.session_state[RUN_KEY]


def set_run(run: RunConfig) -> None:
    st.session_state[RUN_KEY] = run


def reset_run(section: str = "all") -> None:
    cur = get_run()
    if section == "data":
        set_run(dataclasses.replace(cur, data=DataConfig()))
    elif section == "hmm":
        set_run(dataclasses.replace(cur, hmm=HMMConfig()))
    elif section == "bn":
        set_run(dataclasses.replace(cur, bn=BNConfig()))
    else:
        set_run(C.DEFAULT_RUN)


def request_refresh() -> None:
    """Switch to live FRED retrieval and bump the nonce so caches are bypassed."""
    cur = get_run()
    d = dataclasses.replace(cur.data, data_mode="live", refresh_nonce=cur.data.refresh_nonce + 1)
    set_run(dataclasses.replace(cur, data=d))


def _j(obj) -> str:
    return json.dumps(dataclasses.asdict(obj), sort_keys=True, default=list)


def _bn_from_json(s: str) -> BNConfig:
    d = json.loads(s)
    d["indicators"] = tuple(d["indicators"])
    return BNConfig(**d)


# --------------------------------------------------------------------------
# Cached computation
# --------------------------------------------------------------------------

@st.cache_data(ttl=6 * 3600, max_entries=8, show_spinner="Loading and validating data…")
def _cached_dataset(cfg_json: str) -> Dataset:
    return get_dataset(DataConfig(**json.loads(cfg_json)))


@st.cache_resource(max_entries=6, show_spinner="Fitting Gaussian HMM (EM restarts)…")
def _cached_hmm(_ds: Dataset, ds_key: str, hmm_json: str) -> RegimeHMM:
    cfg = HMMConfig(**json.loads(hmm_json))
    return fit_regime_hmm(_ds.features, _ds.X_std, cfg.n_states, cfg.n_restarts, cfg.n_iter,
                          cfg.tol, cfg.covariance_type, vix=_ds.master["VIX"])


@st.cache_resource(max_entries=8, show_spinner="Learning Bayesian-network structure…")
def _cached_bn(_ds: Dataset, _rh: RegimeHMM, ds_key: str, hmm_json: str, bn_json: str):
    cfg = _bn_from_json(bn_json)
    panel = build_discrete_panel(_ds.features, _rh.labels, _ds.master, cfg.discretisation,
                                 cfg.variables, cfg.indicators)
    return panel, build_bn(panel, cfg.max_parents, cfg.score)


def load_dataset(run: Optional[RunConfig] = None) -> Dataset:
    """Return the dataset for the applied configuration or stop the page with a clear error."""
    run = run or get_run()
    try:
        return _cached_dataset(_j(run.data))
    except DataError as exc:
        st.error(f"**Data could not be prepared.** {exc}")
        if st.button("Reset data settings to research defaults", key="err_reset_data"):
            reset_run("data")
            st.rerun()
        st.stop()
        raise  # pragma: no cover


def load_hmm(ds: Dataset, run: Optional[RunConfig] = None) -> RegimeHMM:
    run = run or get_run()
    try:
        return _cached_hmm(ds, ds.fingerprint + _j(run.data), _j(run.hmm))
    except ModelFitError as exc:
        st.error(f"**The HMM did not converge.** {exc}")
        if getattr(exc, "restarts", None) is not None:
            st.dataframe(exc.restarts, width="stretch", hide_index=True)
        if st.button("Reset HMM settings to research defaults", key="err_reset_hmm"):
            reset_run("hmm")
            st.rerun()
        st.stop()
        raise  # pragma: no cover


def load_bn(ds: Dataset, rh: RegimeHMM, run: Optional[RunConfig] = None) -> tuple[pd.DataFrame, BNModel]:
    run = run or get_run()
    key = ds.fingerprint + _j(run.data)
    return _cached_bn(ds, rh, key, _j(run.hmm), _j(run.bn))


@dataclass
class Analysis:
    run: RunConfig
    ds: Dataset
    rh: RegimeHMM
    rule: pd.Series
    panel: pd.DataFrame
    bn: BNModel


def load_analysis(run: Optional[RunConfig] = None) -> Analysis:
    """Everything (dataset, HMM, rule-based benchmark, discrete panel, BN) for the applied run."""
    run = run or get_run()
    ds = load_dataset(run)
    rh = load_hmm(ds, run)
    panel, bn = load_bn(ds, rh, run)
    rule = rule_based_regimes(ds.master).loc[ds.features.index]
    return Analysis(run, ds, rh, rule, panel, bn)


def clear_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
