"""Hidden Markov Model engine — CrudeCortex GWP2.

Part 1 — from-scratch *discrete* HMM algorithms (Forward, Backward, Viterbi,
Baum-Welch) on the report's Sunny/Cloudy/Rainy toy example. Two numerically
distinct implementations are provided:

* textbook probability-space versions (``forward``, ``backward``, ``viterbi``)
  that reproduce the report's tables, and
* log-space versions (``forward_log`` ...) that never underflow and are also
  used to verify the continuous model below.

Part 2 — the 3-state *Gaussian* HMM used for WTI regimes (hmmlearn, full
covariance, up to 200 EM iterations, tol 1e-4, 10 restarts, best converged
log-likelihood retained), with every parameter array re-ordered consistently by
ascending mean return so that labels, matrices and posteriors always agree.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .config import HMMConfig

logging.getLogger("hmmlearn").setLevel(logging.ERROR)   # non-monotone EM notices are reported via the restart table

# --------------------------------------------------------------------------
# Toy example (report Step 1)
# --------------------------------------------------------------------------
TOY_STATES = ["Sunny", "Cloudy", "Rainy"]
TOY_OBS = ["Dry", "Damp", "Wet"]
TOY_OBS_SEQ = [0, 1, 2]                      # Dry, Damp, Wet
TOY_PI = np.array([0.6, 0.3, 0.1])
TOY_A = np.array([[0.7, 0.2, 0.1],
                  [0.3, 0.4, 0.3],
                  [0.2, 0.3, 0.5]])
TOY_B = np.array([[0.5, 0.3, 0.2],
                  [0.2, 0.5, 0.3],
                  [0.2, 0.2, 0.6]])


# --------------------------------------------------------------------------
# Textbook (probability-space) algorithms — report pseudocode
# --------------------------------------------------------------------------

def forward(A, B, pi, obs):
    """alpha_t(j) = P(o1..ot, q_t=j). Returns (alpha T x N, P(O|lambda))."""
    T, N = len(obs), A.shape[0]
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = B[j, obs[t]] * np.sum(alpha[t - 1] * A[:, j])
    return alpha, float(alpha[-1].sum())


def backward(A, B, pi, obs):
    """beta_t(i) = P(o_{t+1}..oT | q_t=i). Returns (beta T x N, P(O|lambda))."""
    T, N = len(obs), A.shape[0]
    beta = np.zeros((T, N))
    beta[T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i] * B[:, obs[t + 1]] * beta[t + 1])
    p = float(np.sum(pi * B[:, obs[0]] * beta[0]))
    return beta, p


def viterbi_trace(A, B, pi, obs):
    """Viterbi with back-pointers. Returns ``(path, delta, psi, log P(Q*, O))``."""
    T, N = len(obs), A.shape[0]
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    delta[0] = pi * B[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            cand = delta[t - 1] * A[:, j]
            psi[t, j] = int(np.argmax(cand))
            delta[t, j] = B[j, obs[t]] * cand.max()
    path = [int(np.argmax(delta[-1]))]
    for t in range(T - 1, 0, -1):
        path.append(int(psi[t, path[-1]]))
    path.reverse()
    with np.errstate(divide="ignore"):
        logp = float(np.log(delta[-1].max()))
    return path, delta, psi, logp


def viterbi(A, B, pi, obs):
    """Most likely hidden path via max + backpointer. Returns (path, delta, logP)."""
    path, delta, _, logp = viterbi_trace(A, B, pi, obs)
    return path, delta, logp


def forward_scaled(A, B, pi, obs):
    """Rabiner-scaled forward pass. Returns (alpha_hat, scale c_t, log P(O|lambda))."""
    T, N = len(obs), A.shape[0]
    ahat = np.zeros((T, N))
    c = np.zeros(T)
    a = pi * B[:, obs[0]]
    c[0] = a.sum()
    ahat[0] = a / c[0]
    for t in range(1, T):
        a = (ahat[t - 1] @ A) * B[:, obs[t]]
        c[t] = a.sum()
        ahat[t] = a / c[t]
    return ahat, c, float(np.sum(np.log(c)))


# --------------------------------------------------------------------------
# Log-space algorithms (generic: toy discrete OR continuous emissions)
# --------------------------------------------------------------------------

def _log(x):
    with np.errstate(divide="ignore"):
        return np.log(x)


def forward_log(log_pi, log_A, log_B):
    """log alpha (T x N) and log P(O|lambda). ``log_B[t, j] = log b_j(o_t)``."""
    T, N = log_B.shape
    la = np.empty((T, N))
    la[0] = log_pi + log_B[0]
    for t in range(1, T):
        la[t] = logsumexp(la[t - 1][:, None] + log_A, axis=0) + log_B[t]
    return la, float(logsumexp(la[-1]))


def backward_log(log_A, log_B):
    """log beta (T x N), with log beta_T = 0."""
    T, N = log_B.shape
    lb = np.zeros((T, N))
    for t in range(T - 2, -1, -1):
        lb[t] = logsumexp(log_A + (log_B[t + 1] + lb[t + 1])[None, :], axis=1)
    return lb


def viterbi_log(log_pi, log_A, log_B):
    """Log-space Viterbi. Returns (path, log delta, best log-probability)."""
    T, N = log_B.shape
    delta = np.empty((T, N))
    psi = np.zeros((T, N), dtype=int)
    delta[0] = log_pi + log_B[0]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + log_A
        psi[t] = np.argmax(cand, axis=0)
        delta[t] = cand[psi[t], np.arange(N)] + log_B[t]
    path = [int(np.argmax(delta[-1]))]
    for t in range(T - 1, 0, -1):
        path.append(int(psi[t, path[-1]]))
    path.reverse()
    return np.array(path), delta, float(delta[-1].max())


def posteriors_log(la, lb, ll):
    """Smoothed state probabilities gamma_t(i) = P(q_t=i | O, lambda)."""
    return np.exp(la + lb - ll)


def gaussian_log_emissions(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """log N(x_t; mu_j, Sigma_j) for every t, j. ``covars`` are full (N, d, d)."""
    from scipy.stats import multivariate_normal
    T, N = X.shape[0], means.shape[0]
    out = np.empty((T, N))
    for j in range(N):
        out[:, j] = multivariate_normal(means[j], covars[j], allow_singular=False).logpdf(X)
    return out


def baum_welch(A, B, pi, obs, n_iter=100, tol=1e-10):
    """Baum-Welch (EM) for a discrete HMM, computed in log-space.

    Returns ``(A, B, pi, ll_hist, gamma)`` where ``ll_hist`` is the
    *log*-likelihood log P(O|lambda) at the start of every EM iteration
    (non-decreasing) and ``gamma`` is the last E-step posterior. States that
    receive no expected counts keep their previous parameter rows.
    """
    obs = np.asarray(obs, dtype=int)
    T, N, M = len(obs), A.shape[0], B.shape[1]
    A, B, pi = A.astype(float).copy(), B.astype(float).copy(), pi.astype(float).copy()
    ll_hist: list[float] = []
    gamma = None
    for it in range(n_iter):
        log_B = _log(B[:, obs].T)
        la, ll = forward_log(_log(pi), _log(A), log_B)
        lb = backward_log(_log(A), log_B)
        if not np.isfinite(ll):
            break
        ll_hist.append(ll)
        gamma = posteriors_log(la, lb, ll)
        log_A = _log(A)
        xi_sum = np.zeros((N, N))
        for t in range(T - 1):
            lx = la[t][:, None] + log_A + (log_B[t + 1] + lb[t + 1])[None, :] - ll
            xi_sum += np.exp(lx)
        # M-step; empty states keep previous rows
        pi = gamma[0].copy()
        denom_a = gamma[:-1].sum(axis=0)
        A = np.where(denom_a[:, None] > 1e-300, xi_sum / np.maximum(denom_a[:, None], 1e-300), A)
        denom_b = gamma.sum(axis=0)
        for k in range(M):
            num = gamma[obs == k].sum(axis=0)
            B[:, k] = np.where(denom_b > 1e-300, num / np.maximum(denom_b, 1e-300), B[:, k])
        if len(ll_hist) > 1 and abs(ll_hist[-1] - ll_hist[-2]) < tol:
            break
    return A, B, pi, ll_hist, gamma


def run_toy_example() -> dict:
    """Full toy run — the report's Sunny/Cloudy/Rainy weather model."""
    A, B, pi, obs = TOY_A, TOY_B, TOY_PI, TOY_OBS_SEQ
    alpha, p_fwd = forward(A, B, pi, obs)
    beta, p_bwd = backward(A, B, pi, obs)
    path, delta, psi, logp = viterbi_trace(A, B, pi, obs)
    ahat, scale, ll_scaled = forward_scaled(A, B, pi, obs)
    Aw, Bw, piw, ll_hist, gamma = baum_welch(A, B, pi, obs)
    # gamma for the ORIGINAL parameters (first E-step), as in the report's Table 8
    log_B = _log(B[:, np.asarray(obs)].T)
    la, ll0 = forward_log(_log(pi), _log(A), log_B)
    lb = backward_log(_log(A), log_B)
    gamma0 = posteriors_log(la, lb, ll0)
    return {
        "states": TOY_STATES, "obs_names": TOY_OBS,
        "alpha": alpha, "beta": beta, "delta": delta, "psi": psi,
        "alpha_scaled": ahat, "scale": scale, "loglik_scaled": ll_scaled,
        "p_forward": p_fwd, "p_backward": p_bwd,
        "path": [TOY_STATES[i] for i in path], "path_idx": path, "logp_viterbi": logp,
        "gamma": gamma, "gamma_initial": gamma0, "ll_hist": ll_hist,
        "p_hist": [float(np.exp(x)) for x in ll_hist],
        "A_learned": Aw, "B_learned": Bw, "pi_learned": piw,
    }


def underflow_demo(T: int = 1500, seed: int = 7) -> dict:
    """Show why log-space matters: simulate a long toy sequence and compare methods."""
    rng = np.random.default_rng(seed)
    obs, q = [], rng.choice(3, p=TOY_PI)
    for _ in range(T):
        obs.append(int(rng.choice(3, p=TOY_B[q])))
        q = rng.choice(3, p=TOY_A[q])
    obs = np.array(obs)
    _, p_plain = forward(TOY_A, TOY_B, TOY_PI, obs)
    _, _, ll_scaled = forward_scaled(TOY_A, TOY_B, TOY_PI, obs)
    _, ll_log = forward_log(_log(TOY_PI), _log(TOY_A), _log(TOY_B[:, obs].T))
    return {"T": T, "p_unscaled": p_plain, "loglik_scaled": ll_scaled, "loglik_log": ll_log}


# --------------------------------------------------------------------------
# Gaussian HMM for WTI regimes
# --------------------------------------------------------------------------

def regime_names(n_states: int) -> list[str]:
    """Labels in ascending order of mean standardised return."""
    if n_states == 3:
        return ["Bear", "Stagnant", "Bull"]
    if n_states == 2:
        return ["Bear", "Bull"]
    return [f"State {i + 1}" for i in range(n_states)]


class ModelFitError(RuntimeError):
    """Raised when no EM restart converged; carries the restart table."""

    def __init__(self, message: str, restarts: Optional[pd.DataFrame] = None):
        super().__init__(message)
        self.restarts = restarts


@dataclass
class RegimeHMM:
    # --- original contract -------------------------------------------------
    model: object                    # best hmmlearn GaussianHMM (raw state order)
    labels: pd.Series                # Viterbi-decoded regime name per month
    ll: float                        # best log-likelihood
    transmat: np.ndarray             # transition matrix, rows/cols in `state_names` order
    means_std: np.ndarray            # state means in standardised units, `state_names` order
    state_stats: pd.DataFrame        # empirical per-regime statistics
    # --- extended ------------------------------------------------------------
    config: HMMConfig = field(default_factory=HMMConfig)
    state_names: list[str] = field(default_factory=list)
    order: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))   # raw index of each ordered state
    startprob: np.ndarray = field(default_factory=lambda: np.array([]))
    covars_std: np.ndarray = field(default_factory=lambda: np.array([]))
    means_orig: np.ndarray = field(default_factory=lambda: np.array([]))
    covars_orig: np.ndarray = field(default_factory=lambda: np.array([]))
    posteriors: pd.DataFrame = field(default_factory=pd.DataFrame)
    restarts: pd.DataFrame = field(default_factory=pd.DataFrame)
    convergence: list[float] = field(default_factory=list)
    n_iter_used: int = 0
    converged: bool = True
    n_params: int = 0
    aic: float = float("nan")
    bic: float = float("nan")
    spells: pd.DataFrame = field(default_factory=pd.DataFrame)
    duration_summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def raw_index_map(self) -> dict[str, int]:
        return {n: int(i) for n, i in zip(self.state_names, self.order)}


def _n_params(n: int, d: int, cov: str) -> int:
    cov_p = {"full": n * d * (d + 1) // 2, "diag": n * d, "spherical": n, "tied": d * (d + 1) // 2}[cov]
    return (n - 1) + n * (n - 1) + n * d + cov_p


def compute_spells(labels: pd.Series) -> pd.DataFrame:
    """Consecutive-month runs of each regime (a 'spell')."""
    arr = labels.to_numpy()
    rows, start = [], 0
    for i in range(1, len(arr) + 1):
        if i == len(arr) or arr[i] != arr[start]:
            rows.append({"Regime": arr[start], "Start": labels.index[start],
                         "End": labels.index[i - 1], "Months": i - start})
            start = i
    return pd.DataFrame(rows)


def duration_summary(spells: pd.DataFrame, transmat: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Observed spell lengths vs the model-implied expected duration 1/(1-a_ii)."""
    out = []
    for i, n in enumerate(names):
        s = spells.loc[spells["Regime"] == n, "Months"]
        aii = float(transmat[i, i])
        out.append({
            "Regime": n, "Spells": int(len(s)),
            "Mean spell (months)": float(s.mean()) if len(s) else np.nan,
            "Median": float(s.median()) if len(s) else np.nan,
            "Longest": int(s.max()) if len(s) else 0,
            "Shortest": int(s.min()) if len(s) else 0,
            "Self-transition a_ii": aii,
            "Model expected duration 1/(1-a_ii)": (1.0 / (1.0 - aii)) if aii < 1 else float("inf"),
        })
    return pd.DataFrame(out).set_index("Regime")


class _RecordingMonitor:
    """hmmlearn keeps only the last 2 log-likelihoods; this wrapper records them all."""

    @staticmethod
    def attach(model):
        from hmmlearn.base import ConvergenceMonitor

        class Rec(ConvergenceMonitor):
            all_history: list

            def _reset(self):
                super()._reset()
                self.all_history = []

            def report(self, log_prob):
                if not hasattr(self, "all_history"):
                    self.all_history = []
                self.all_history.append(float(log_prob))
                super().report(log_prob)

        model.monitor_ = Rec(model.tol, model.n_iter, model.verbose)
        model.monitor_.all_history = []
        return model


def _fit_one(X: np.ndarray, cfg: HMMConfig, seed: int):
    from hmmlearn.hmm import GaussianHMM
    m = GaussianHMM(n_components=cfg.n_states, covariance_type=cfg.covariance_type,
                    n_iter=cfg.n_iter, tol=cfg.tol, random_state=seed)
    _RecordingMonitor.attach(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X)
        ll = float(m.score(X))
    return m, ll


def fit_regime_hmm(feature_df: pd.DataFrame, X_std: np.ndarray,
                   n_states: int = 3, n_restarts: int = 10,
                   n_iter: int = 200, tol: float = 1e-4,
                   covariance_type: str = "full",
                   vix: Optional[pd.Series] = None) -> RegimeHMM:
    """Fit the Gaussian HMM with random restarts; keep the best *converged* run.

    Seeds are ``0 .. n_restarts-1`` (as in the notebook). Raises
    :class:`ModelFitError` — with the restart table attached — when no restart
    converges, instead of silently returning an unconverged model.
    """
    cfg = HMMConfig(n_states, n_restarts, n_iter, tol, covariance_type)  # type: ignore[arg-type]
    best_model, best_ll, best_seed = None, -np.inf, -1
    rows = []
    for seed in range(cfg.n_restarts):
        try:
            m, ll = _fit_one(X_std, cfg, seed)
        except Exception as exc:  # degenerate covariance etc.
            rows.append({"Seed": seed, "Converged": False, "EM iterations": 0,
                         "Log-likelihood": np.nan, "Note": f"failed: {type(exc).__name__}"})
            continue
        conv = bool(m.monitor_.converged)
        rows.append({"Seed": seed, "Converged": conv, "EM iterations": int(m.monitor_.iter),
                     "Log-likelihood": ll, "Note": "" if conv else "not converged"})
        if conv and ll > best_ll:
            best_ll, best_model, best_seed = ll, m, seed
    restarts = pd.DataFrame(rows)
    if best_model is None:
        raise ModelFitError(
            "None of the EM restarts converged. Increase the maximum iterations, relax the "
            "tolerance, add restarts or reduce the number of states.", restarts)
    restarts["Selected"] = restarts["Seed"] == best_seed

    m = best_model
    N, d = cfg.n_states, X_std.shape[1]
    order = np.argsort(m.means_[:, 0])                 # ascending mean (standardised) return
    names = regime_names(N)
    name_of_raw = {int(order[k]): names[k] for k in range(N)}
    hidden = m.predict(X_std)                          # Viterbi decoding
    labels = pd.Series([name_of_raw[int(h)] for h in hidden], index=feature_df.index, name="Regime")

    A_ord = m.transmat_[np.ix_(order, order)].copy()
    pi_ord = m.startprob_[order].copy()
    means_std = m.means_[order].copy()
    covars_std = np.asarray(m.covars_)[order].copy()
    scaler_mean = feature_df.mean().to_numpy()
    scaler_scale = feature_df.std(ddof=0).to_numpy()
    means_orig = means_std * scaler_scale + scaler_mean
    covars_orig = covars_std * np.outer(scaler_scale, scaler_scale)[None, :, :]
    post = pd.DataFrame(m.predict_proba(X_std)[:, order], index=feature_df.index, columns=names)

    raw = feature_df.copy()
    raw["Regime"] = labels
    if vix is not None:
        raw["VIX"] = vix.reindex(feature_df.index)
    agg = {"Months": ("WTI_ret", "size"),
           "MeanReturn": ("WTI_ret", lambda x: float(np.mean(x)) * 100),
           "MeanVol": ("WTI_vol12", lambda x: float(np.mean(x)) * 100)}
    if vix is not None:
        agg["MeanVIX"] = ("VIX", "mean")
    stats = raw.groupby("Regime").agg(**agg).reindex(names)
    stats["Months"] = stats["Months"].fillna(0).astype(int)
    stats["PctSample"] = stats["Months"] / stats["Months"].sum() * 100

    spells = compute_spells(labels)
    npar = _n_params(N, d, cfg.covariance_type)
    T = len(X_std)
    return RegimeHMM(
        model=m, labels=labels, ll=float(best_ll), transmat=A_ord, means_std=means_std,
        state_stats=stats, config=cfg, state_names=names, order=order, startprob=pi_ord,
        covars_std=covars_std, means_orig=means_orig, covars_orig=covars_orig,
        posteriors=post, restarts=restarts, convergence=list(m.monitor_.all_history),
        n_iter_used=int(m.monitor_.iter), converged=bool(m.monitor_.converged),
        n_params=npar, aic=float(2 * npar - 2 * best_ll),
        bic=float(npar * np.log(T) - 2 * best_ll),
        spells=spells, duration_summary=duration_summary(spells, A_ord, names))


def verify_gaussian_hmm(rh: RegimeHMM, X_std: np.ndarray) -> dict:
    """Independent log-space re-computation of hmmlearn's likelihood and Viterbi path.

    Uses the *raw* hmmlearn parameters and our own ``forward_log`` / ``viterbi_log``.
    Returns the largest discrepancies — they should be ~1e-10.
    """
    m = rh.model
    covars = np.asarray(m.covars_)
    log_B = gaussian_log_emissions(X_std, m.means_, covars)
    log_pi, log_A = _log(m.startprob_), _log(m.transmat_)
    la, ll = forward_log(log_pi, log_A, log_B)
    lb = backward_log(log_A, log_B)
    ll_b = float(logsumexp(log_pi + log_B[0] + lb[0]))
    path, _, _ = viterbi_log(log_pi, log_A, log_B)
    gamma = posteriors_log(la, lb, ll)
    return {
        "loglik_own_total": ll, "loglik_hmmlearn_total": float(m.score(X_std)),
        "loglik_abs_gap": abs(ll - float(m.score(X_std))),
        "forward_backward_gap": abs(ll - ll_b),
        "viterbi_agreement": float(np.mean(path == m.predict(X_std))),
        "posterior_max_gap": float(np.max(np.abs(gamma - m.predict_proba(X_std)))),
        "gamma_rows_sum_to_one": bool(np.allclose(gamma.sum(axis=1), 1.0)),
    }


# --------------------------------------------------------------------------
# Rule-based benchmark (report Step 2)
# --------------------------------------------------------------------------

def rule_based_regimes(df: pd.DataFrame, bull_ret: float = 0.10, bull_vol: float = 0.40,
                       bear_ret: float = -0.10, bear_vol: float = 0.30) -> pd.Series:
    """Report Step 2 benchmark: trailing-12M return and annualised-volatility gates.

    Bull: 12M return > +10% and volatility < 40%.  Bear: 12M return < -10% and
    volatility > 30%.  Everything else: Stagnant.
    """
    ret12 = df["WTI"].pct_change(12)
    vol12 = np.log(df["WTI"] / df["WTI"].shift(1)).rolling(12).std() * np.sqrt(12)
    regimes = pd.Series("Stagnant", index=df.index, name="RuleRegime")
    regimes[(ret12 > bull_ret) & (vol12 < bull_vol)] = "Bull"
    regimes[(ret12 < bear_ret) & (vol12 > bear_vol)] = "Bear"
    return regimes


def regime_summary(df: pd.DataFrame, regimes: pd.Series,
                   vol12: pd.Series | None = None) -> pd.DataFrame:
    """Per-regime summary. Masks are reindexed to df.index so a 216-month
    regime Series can be summarised against the 228-month master panel."""
    reg = regimes.reindex(df.index)
    if vol12 is None:
        vol12 = np.log(df["WTI"] / df["WTI"].shift(1)).rolling(12).std() * np.sqrt(12)
    else:
        vol12 = vol12.reindex(df.index)
    ret12 = df["WTI"].pct_change(12)          # report's trailing 12M return
    rows = []
    for reg_name in ["Bull", "Stagnant", "Bear"]:
        m = (reg == reg_name).fillna(False)
        if m.sum() == 0:
            continue
        rows.append({
            "Regime": reg_name, "Months": int(m.sum()),
            "PctSample": round(float(m.mean() * 100), 1),
            "MeanWTI": round(float(df.loc[m, "WTI"].mean()), 2),
            "MeanMonthlyReturn": round(float(ret12[m].mean() * 100 / 12), 2),
            "MeanAnnVol": round(float(vol12[m].mean() * 100), 1),
        })
    return pd.DataFrame(rows).set_index("Regime")


def regime_agreement(hmm_labels: pd.Series, rule_labels: pd.Series) -> pd.DataFrame:
    """Cross-tabulate HMM-decoded regimes against the rule-based benchmark (descriptive)."""
    r = rule_labels.reindex(hmm_labels.index)
    return pd.crosstab(hmm_labels.rename("HMM (rows)"), r.rename("Rule-based (columns)"))
