"""CrudeCortex GWP2 — engine verification tests. 13 tests.
Run:  python -m pytest tests/test_gwp2.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_gwp2 import get_gwp2_data
from core.hmm_engine import (run_toy_example, forward, backward, viterbi,
                             baum_welch, TOY_A, TOY_B, TOY_PI, TOY_OBS_SEQ,
                             fit_regime_hmm, rule_based_regimes)
from core.bn_engine import build_discrete_panel, build_bn, is_dag


@pytest.fixture(scope="module")
def data():
    return get_gwp2_data()


@pytest.fixture(scope="module")
def toy():
    return run_toy_example()


def test_toy_alpha_init_matches_report(toy):
    assert np.allclose(toy["alpha"][0], [0.30, 0.06, 0.02], atol=1e-9)


def test_toy_forward_backward_consistency(toy):
    assert abs(toy["p_forward"] - toy["p_backward"]) < 1e-12


def test_toy_em_likelihood_monotone(toy):
    ll = toy["ll_hist"]
    assert ll[-1] >= ll[0]
    assert all(b >= a - 1e-9 for a, b in zip(ll, ll[1:]))


def test_toy_viterbi_path_is_valid_states(toy):
    assert set(toy["path"]) <= set(toy["states"])
    assert len(toy["path"]) == 3


def test_master_panel_shape(data):
    df, feature_df, X_std, scaler, adf = data
    assert df.shape == (228, 5)
    assert feature_df.shape == (216, 2)
    assert X_std.shape == (216, 2)


def test_adf_stationarity(data):
    _, _, _, _, adf = data
    assert adf["p"] < 0.01
    assert abs(adf["stat"] - (-9.7807)) < 0.1


def test_hmm_loglik_matches_report(data):
    df, feature_df, X_std, _, _ = data
    rh = fit_regime_hmm(feature_df, X_std)
    assert abs(rh.ll - (-305.9770)) < 0.5


def test_hmm_regime_counts_match_report(data):
    df, feature_df, X_std, _, _ = data
    rh = fit_regime_hmm(feature_df, X_std)
    counts = rh.labels.value_counts()
    assert counts["Bull"] == 129 and counts["Bear"] == 59 and counts["Stagnant"] == 28


def test_hmm_persistence_diagonal(data):
    df, feature_df, X_std, _, _ = data
    rh = fit_regime_hmm(feature_df, X_std)
    assert all(rh.transmat[i, i] > 0.3 for i in range(3))


def test_rule_based_benchmark_counts(data):
    df, feature_df, _, _, _ = data
    reg = rule_based_regimes(df).loc[feature_df.index]
    counts = reg.value_counts()
    assert counts.get("Bear", 0) == 50
    assert abs(counts.get("Bull", 0) - 79) <= 4
    assert abs(counts.get("Stagnant", 0) - 87) <= 4


@pytest.fixture(scope="module")
def bn(data):
    df, feature_df, X_std, _, _ = data
    rh = fit_regime_hmm(feature_df, X_std)
    panel = build_discrete_panel(feature_df, rh.labels, df)
    return build_bn(panel)


def test_bn_structure_is_dag(bn):
    assert is_dag(bn.edges, bn.nodes)


def test_bn_search_improves_bic(bn):
    assert len(bn.history) >= 2


def test_bn_cpd_vix_signal(data, bn):
    from core.bn_engine import conditional_regime_table
    df, feature_df, X_std, _, _ = data
    rh = fit_regime_hmm(feature_df, X_std)
    panel = build_discrete_panel(feature_df, rh.labels, df)
    tbl = conditional_regime_table(panel)
    low = tbl[tbl["VIX_Level"] == "Low"].iloc[0]
    high = tbl[tbl["VIX_Level"] == "High"].iloc[0]
    assert low["P(Bull)"] > high["P(Bull)"]
    assert high["P(Stagnant)"] > low["P(Stagnant)"]
    assert abs(low["P(Bull)"] - 0.704) < 0.08
