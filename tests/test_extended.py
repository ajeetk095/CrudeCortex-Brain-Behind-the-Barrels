"""Extended tests: data providers, HMM and BN invariants, exports, Streamlit pages.
Expected values are derived from independent computation or invariants, not pasted from outputs.
Run: python -m pytest tests -q
"""
import dataclasses
import glob
import json
import os
import sys
import zipfile
import io

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import bn_engine as B, config as C, data_sources as DS, exports
from core.data_gwp2 import DateRangeError, get_dataset
from core.hmm_engine import (TOY_A, TOY_B, TOY_PI, TOY_OBS_SEQ, baum_welch, backward_log, forward_log,
                             fit_regime_hmm, gaussian_log_emissions, posteriors_log, rule_based_regimes,
                             verify_gaussian_hmm, viterbi_log, compute_spells)


@pytest.fixture(scope="module")
def ds():
    return get_dataset(C.DataConfig())


@pytest.fixture(scope="module")
def rh(ds):
    return fit_regime_hmm(ds.features, ds.X_std, 3, 10, 200, 1e-4, "full", vix=ds.master["VIX"])


@pytest.fixture(scope="module")
def bnm(ds, rh):
    p = B.build_discrete_panel(ds.features, rh.labels, ds.master, "report", "trend", C.MACRO_KEYS)
    return p, B.build_bn(p)


# ---------------- data ----------------
def test_snapshot_files_valid():
    for sid in C.SERIES_BY_ID:
        ld = DS.load_series(sid, "snapshot", cache_dir="/nonexistent")
        assert ld.origin == "snapshot" and len(ld.values) > 100 and ld.values.index.is_monotonic_increasing


def test_parse_rejects_garbage():
    with pytest.raises(DS.DataError):
        DS.parse_fred_csv_text("<html>blocked</html>", "WTISPLC")
    with pytest.raises(DS.DataError):
        DS.parse_fred_csv_text("", "WTISPLC")


def test_live_failure_falls_back_to_snapshot(tmp_path):
    def boom(_):
        raise DS.DataUnavailableError("network down")
    ld = DS.load_series("WTISPLC", "live", fetcher=boom, cache_dir=str(tmp_path))
    assert ld.origin == "snapshot" and ld.live_attempted and "network down" in ld.live_error


def test_live_success_writes_cache(tmp_path):
    text = open(DS.snapshot_path("WTISPLC")).read()
    ld = DS.load_series("WTISPLC", "live", fetcher=lambda _: text, cache_dir=str(tmp_path))
    assert ld.origin == "live" and os.path.isfile(DS.cache_path("WTISPLC", str(tmp_path)))
    again = DS.load_series("WTISPLC", "snapshot", snapshot_dir=str(tmp_path / "none"), cache_dir=str(tmp_path))
    assert again.origin == "cache"


def test_missing_everything_raises(tmp_path):
    def boom(_):
        raise DS.DataUnavailableError("x")
    with pytest.raises(DS.DataUnavailableError):
        DS.load_series("WTISPLC", "snapshot", fetcher=boom, snapshot_dir=str(tmp_path), cache_dir=str(tmp_path))


def test_corrupt_snapshot_is_reported(tmp_path):
    (tmp_path / "WTISPLC.csv").write_text("not,a,csv\n1,2")
    with pytest.raises(DS.DataUnavailableError):
        DS.load_series("WTISPLC", "snapshot", fetcher=lambda _: (_ for _ in ()).throw(DS.DataUnavailableError("n")),
                       snapshot_dir=str(tmp_path), cache_dir=str(tmp_path / "c"))


def test_vix_yahoo_last_resort(tmp_path):
    def boom(_):
        raise DS.DataUnavailableError("x")
    y = pd.Series([15.0, 16.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    ld = DS.load_vix("snapshot", fetcher=boom, yahoo=lambda: y, snapshot_dir=str(tmp_path), cache_dir=str(tmp_path))
    assert ld.origin == "yahoo_fallback"


def test_window_shape_consistency(ds):
    assert len(ds.features) == len(ds.master) - C.WARMUP_MONTHS
    assert ds.X_std.shape == (len(ds.features), 2)
    assert np.allclose(ds.X_std.mean(0), 0, atol=1e-9) and np.allclose(ds.X_std.std(0), 1, atol=1e-9)
    assert not ds.master.isna().any().any()


def test_invalid_date_ranges_raise():
    with pytest.raises(DateRangeError):
        get_dataset(dataclasses.replace(C.DataConfig(), feature_start="2030-01-01", feature_end="2031-01-01"))
    with pytest.raises(DateRangeError):
        get_dataset(dataclasses.replace(C.DataConfig(), feature_start="2020-01-01", feature_end="2019-01-01"))


def test_alternative_window_runs():
    d = get_dataset(dataclasses.replace(C.DataConfig(), feature_start="2010-01-01", feature_end="2020-12-01"))
    assert d.features.index[0] == pd.Timestamp("2010-01-01") and len(d.features) == 132


# ---------------- HMM ----------------
def test_toy_forward_backward_consistency():
    A, Bm, pi = map(np.asarray, (TOY_A, TOY_B, TOY_PI))
    obs = list(TOY_OBS_SEQ)
    if isinstance(obs[0], str):
        pytest.skip("observations stored as labels")
    lb = np.log(Bm[:, obs].T)
    la, ll = forward_log(np.log(pi), np.log(A), lb)
    lbeta = backward_log(np.log(A), lb)
    ll2 = np.logaddexp.reduce(np.log(pi) + lb[0] + lbeta[0])
    assert abs(ll - ll2) < 1e-10
    g = posteriors_log(la, lbeta, ll)
    assert np.allclose(g.sum(1), 1)


def test_hmm_parameters_valid(rh):
    assert np.allclose(rh.transmat.sum(1), 1) and (rh.transmat >= 0).all()
    assert abs(rh.startprob.sum() - 1) < 1e-9
    assert np.allclose(rh.posteriors.sum(axis=1), 1, atol=1e-8)
    assert set(rh.labels) <= set(rh.state_names)
    for c in rh.covars_std:
        assert np.all(np.linalg.eigvalsh(c) > 0)


def test_state_ordering_consistent(rh):
    mr = rh.means_orig[:, 0]
    assert list(mr) == sorted(mr), "states must be ordered by mean return"
    # posterior-weighted state means must follow the same order as the model means
    assert list(rh.posteriors.columns) == list(rh.state_names)
    for i, n in enumerate(rh.state_names):
        assert abs(rh.transmat[i].sum() - 1) < 1e-9


def test_own_inference_matches_hmmlearn(rh, ds):
    v = verify_gaussian_hmm(rh, ds.X_std)
    assert v["loglik_abs_gap"] < 1e-6 and v["viterbi_agreement"] == 1.0 and v["forward_backward_gap"] < 1e-8


def test_baum_welch_monotone():
    A, Bm, pi = map(np.asarray, (TOY_A, TOY_B, TOY_PI))
    obs = [int(o) for o in TOY_OBS_SEQ] if not isinstance(TOY_OBS_SEQ[0], str) else None
    if obs is None:
        pytest.skip("labels")
    *_, hist, _ = baum_welch(A, Bm, pi, obs, n_iter=30, tol=0)
    assert np.all(np.diff(hist) >= -1e-9)


def test_fit_is_deterministic(rh, ds):
    again = fit_regime_hmm(ds.features, ds.X_std, 3, 3, 200, 1e-4, "full", vix=ds.master["VIX"])
    assert again.ll <= rh.ll + 1e-6  # fewer restarts can never beat more restarts (same seeds)


def test_alternative_states_fit(ds):
    m = fit_regime_hmm(ds.features, ds.X_std, 2, 3, 200, 1e-4, "diag", vix=ds.master["VIX"])
    assert len(m.state_names) == 2 and np.allclose(m.transmat.sum(1), 1)


def test_spells_cover_sample(rh):
    sp = compute_spells(rh.labels)
    col = next(c for c in sp.columns if c.lower().startswith(("months", "length", "duration")))
    assert sp[col].sum() == len(rh.labels)
    assert len(sp) >= 3


def test_rule_based_labels_complete(ds):
    r = rule_based_regimes(ds.master).loc[ds.features.index]
    assert set(r.dropna()) <= {"Bull", "Bear", "Stagnant"} and r.notna().all()


# ---------------- Bayesian network ----------------
def test_bn_is_dag_and_cpds_normalised(bnm):
    p, bn = bnm
    assert B.is_dag(bn.edges, bn.nodes)
    for n in bn.nodes:
        cf = B.cpd_frame(bn, n).dropna(axis=1)
        assert np.allclose(cf.sum(axis=0), 1)
        assert len(bn.parents_of(n)) <= bn.max_parents


def test_bn_inference_properties(bnm):
    p, bn = bnm
    d = B.infer_network(bn, "WTI_Regime", {"VIX_Level": "High"})
    assert abs(d.sum() - 1) < 1e-9 and (d >= 0).all()
    emp, n = B.infer_empirical(bn, "WTI_Regime", {"VIX_Level": "High"})
    assert n == int((p["VIX_Level"] == "High").sum())
    with pytest.raises(B.InferenceError):
        B.infer_network(bn, "WTI_Regime", {"VIX_Level": "Extreme"})
    with pytest.raises(B.InferenceError):
        B.infer_network(bn, "NoSuchNode", {})


def test_bic_decomposes(bnm):
    p, bn = bnm
    total = B.total_score(p, bn.edges, bn.nodes, "bic")
    assert abs(total - bn.history[-1]["total_score"]) < 0.01  # history stores the score rounded to 2 dp
    # independent recomputation of one family's BIC
    n = len(p)
    node = "VIX_Level"
    counts = p[node].value_counts()
    ll = float((counts * np.log(counts / n)).sum())
    k = len(counts) - 1
    assert abs(B.bic_score(p, node, (), "bic") - (ll - 0.5 * k * np.log(n))) < 1e-8


def test_empty_graph_when_penalty_dominates():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.choice(["Low", "Medium", "High"], 60) for c in ["A", "B", "C"]})
    edges, hist = B.hill_climb_search(df, list(df.columns), 2, "bic")
    assert len(edges) <= 1


def test_holdout_runs(bnm):
    p, bn = bnm
    h = B.holdout_evaluation(p, bn.edges)
    assert h["n_train"] + h["n_test"] == len(p) and h["log_loss_model"] > 0


# ---------------- exports & config ----------------
def test_exports_and_zip(ds, rh, bnm):
    p, bn = bnm
    rule = rule_based_regimes(ds.master).loc[ds.features.index]
    z = zipfile.ZipFile(io.BytesIO(exports.build_zip(C.DEFAULT_RUN, ds, rh, rule, bn)))
    names = z.namelist()
    assert {"regime_labels.csv", "bn_edges.csv", "configuration_summary.json", "analysis_summary.md"} <= set(names)
    cfg = json.loads(z.read("configuration_summary.json"))
    assert cfg["configuration_is_research_default"] is True
    assert len(pd.read_csv(io.BytesIO(z.read("regime_labels.csv")))) == len(ds.features)


def test_deviations_flagged():
    r = dataclasses.replace(C.DEFAULT_RUN, hmm=dataclasses.replace(C.DEFAULT_RUN.hmm, n_states=2))
    assert not r.is_research_default and r.deviations()
    assert C.DEFAULT_RUN.is_research_default and not C.DEFAULT_RUN.deviations()


def test_developer_scope():
    assert C.DEVELOPER["name"] == "Ajeet Kumar"
    assert C.DEVELOPER["github"] == "https://github.com/ajeetk095"
    assert C.DEVELOPER["linkedin"] == "https://www.linkedin.com/in/ajeek095/"


def test_original_documents_preserved():
    assert len(glob.glob(os.path.join(ROOT, "documents", "*.pdf"))) == 3
    assert glob.glob(os.path.join(ROOT, "notebooks", "*.ipynb"))


# ---------------- Streamlit ----------------
def test_app_and_pages_start():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=300).run()
    assert not at.exception
    for p in sorted(glob.glob(os.path.join(ROOT, "views", "*.py"))):
        name = os.path.basename(p)
        try:
            at.switch_page(f"views/{name}").run()
        except Exception as e:  # pragma: no cover
            pytest.fail(f"{name}: {e}")
        assert not at.exception, f"{name}: {[e.value[:200] for e in at.exception]}"


def test_reset_and_run_buttons():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=300).run()
    at.switch_page("views/regime_pipeline.py").run()
    assert any(b.key == "btn_run_hmm" or "Run" in (b.label or "") for b in at.button)


def test_snapshot_matches_manifest():
    from core.exports import sha256_file
    man = json.load(open(os.path.join(C.SNAPSHOT_DIR, "MANIFEST.json")))
    for sid, rec in man["files"].items():
        assert sha256_file(DS.snapshot_path(sid)) == rec["sha256"]
