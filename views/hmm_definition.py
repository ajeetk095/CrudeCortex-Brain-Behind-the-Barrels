"""HMM Definition & Design — formal model, specification, labelling logic and research design (Steps 3–4)."""
import pandas as pd
import streamlit as st

from core import config as C
from core import state, ui_components as ui

run = state.get_run()
h, d0 = run.hmm, C.HMMConfig()

ui.page_header("HMM Definition & Design", "The formal model behind the crude-oil regimes, the exact specification in use, and how the research is organised.",
               "HMM research · Steps 3–4")
ui.config_banner(run)

ui.section("Formal definition")
ui.dual_explanation(
    "The market is assumed to be in one of a few hidden phases each month. Each phase tends to last a while, and each phase produces typical-looking returns and volatility. "
    "We cannot see the phase, but we can estimate it from the returns and volatility we do see.",
    "A Hidden Markov Model is λ = (π, A, {μ_j, Σ_j}) with hidden states q_t ∈ {1,…,N}:")
st.latex(r"P(q_1=i)=\pi_i,\qquad P(q_{t+1}=j\mid q_t=i)=a_{ij},\qquad \mathbf{x}_t\mid q_t=j\;\sim\;\mathcal N(\boldsymbol\mu_j,\Sigma_j)")
st.latex(r"\mathbf{x}_t=\big(z[r_t],\,z[\sigma^{12}_t]\big),\quad r_t=\ln\frac{P_t}{P_{t-1}},\quad \sigma^{12}_t=\sqrt{12}\cdot\mathrm{std}(r_{t-11},\dots,r_t)")
st.latex(r"\ln P(X\mid\lambda)=\ln\sum_{q_1..q_T}\pi_{q_1}b_{q_1}(\mathbf x_1)\prod_{t=2}^{T}a_{q_{t-1}q_t}b_{q_t}(\mathbf x_t)")
st.markdown("**Assumptions:** first-order Markov regime process; Gaussian emissions given the regime (so fat tails are captured only through the mixture of states); "
            "time-homogeneous transition probabilities; features standardised (z-scores) over the modelled sample.")

ui.section("Specification in use")
rows = [("Hidden states", h.n_states, d0.n_states, "Bull / Bear / Stagnant"),
        ("Observed features", "WTI log-return, 12M annualised volatility", "same", "Both standardised"),
        ("Emission distribution", f"Gaussian, {h.covariance_type} covariance", f"Gaussian, {d0.covariance_type} covariance", "Full = features may correlate within a state"),
        ("Estimation", "Baum–Welch (EM)", "same", "hmmlearn GaussianHMM"),
        ("Max EM iterations", h.n_iter, d0.n_iter, ""),
        ("Convergence tolerance", f"{h.tol:g}", f"{d0.tol:g}", "Change in log-likelihood per iteration"),
        ("Random restarts", h.n_restarts, d0.n_restarts, "Seeds 0 … n−1; best converged log-likelihood kept"),
        ("Decoding", "Viterbi (most likely path)", "same", "Smoothed posteriors are shown separately")]
st.dataframe(pd.DataFrame(rows, columns=["Setting", "Current run", "Report", "Comment"]).astype(str), width="stretch", hide_index=True)
if not run.hmm == d0:
    st.warning("The current HMM settings differ from the submitted report; results on the Pipeline page are an **alternative model run**.")
st.caption("Change these on the Regime Detection Pipeline page.")

ui.section("State labelling logic")
ui.dual_explanation(
    "The model itself only knows 'state 0, 1, 2'. To make them readable, the state with the lowest average return is called Bear, the highest Bull, and the one in between Stagnant. "
    "Because that is only an ordering, always check each state's volatility and behaviour before trusting a name.",
    "After fitting, states are sorted by the mean of the **standardised return** component, μ_j[0]. The same permutation is applied to the initial vector π, both axes of A, "
    "the means, the covariance matrices and the posterior-probability columns, so every displayed table refers to the same named state. "
    "The raw hmmlearn index of each label is shown on the Pipeline page for reconciliation with the report (where raw state 0 = Bear, 1 = Bull, 2 = Stagnant). "
    "For two states the labels are Bear/Bull; for four or five states, generic names 'State 1 … State n' (lowest to highest mean return) are used.")

ui.section("Research design (Step 4)")
ui.cards([
    ("1 · Macro research", "Choose indicators with a plausible link to oil: market fear (VIX), industrial activity, energy inflation and the dollar."),
    ("2 · Regime process", "Fit the three-state Gaussian HMM to WTI returns and volatility; decode the most likely regime for every month."),
    ("3 · Bayesian-network training", "Discretise the macro indicators and the regime, learn the dependency structure by Hill Climb Search with BIC and estimate CPDs."),
], per_row=3)
ui.disclaimer()
