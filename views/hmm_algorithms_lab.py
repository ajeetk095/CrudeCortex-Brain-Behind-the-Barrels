"""HMM Algorithms Lab — Forward, Backward, Viterbi, Baum–Welch on the Sunny/Cloudy/Rainy toy model."""
import numpy as np
import pandas as pd
import streamlit as st

from core import charts as G
from core import ui_components as ui
from core.hmm_engine import (TOY_A, TOY_B, TOY_OBS, TOY_OBS_SEQ, TOY_PI, TOY_STATES, run_toy_example, underflow_demo)


@st.cache_data(show_spinner=False)
def _toy():
    return run_toy_example()


@st.cache_data(show_spinner=False)
def _underflow(T: int):
    return underflow_demo(T)


toy = _toy()
S, O = TOY_STATES, TOY_OBS
obs_names = [O[i] for i in TOY_OBS_SEQ]
tcols = [f"t={t + 1} ({o})" for t, o in enumerate(obs_names)]

ui.page_header("HMM Algorithms Lab", "The four classic HMM algorithms, implemented from scratch and demonstrated on a small discrete weather model.",
               "HMM research · Step 1")
ui.callout("warn", "Discrete toy model vs. the real crude-oil model",
           "This lab uses a **discrete** HMM (hidden weather states, categorical observations) because every number can be checked by hand. "
           "The crude-oil regimes use a **continuous Gaussian** HMM: the observation probability b_j(o) becomes a bivariate normal density "
           "of (log-return, volatility). The recursions are identical; only the emission term changes. The same log-space routines are used to verify the Gaussian model on the Pipeline page.")

tabs = st.tabs(["Model", "Forward", "Backward", "Viterbi", "Baum–Welch", "Stability & complexity"])

with tabs[0]:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Hidden states** " + ", ".join(S))
        st.markdown("**Observations** " + ", ".join(O))
        st.markdown(f"**Observed sequence** {' → '.join(obs_names)}")
        st.markdown("**Initial probabilities π**")
        st.dataframe(pd.DataFrame([TOY_PI], columns=S, index=["π"]).round(3), width="stretch")
    with c2:
        st.markdown("**Transition matrix A** (row = current state, column = next state)")
        st.dataframe(pd.DataFrame(TOY_A, index=S, columns=S).round(3), width="stretch")
        st.markdown("**Emission matrix B** (row = state, column = observation)")
        st.dataframe(pd.DataFrame(TOY_B, index=S, columns=O).round(3), width="stretch")
    ui.dual_explanation(
        "Imagine we cannot see the weather, only whether the ground is dry, damp or wet. Each kind of weather makes each ground condition more or less likely, "
        "and the weather tends to persist from day to day. An HMM puts numbers on exactly this.",
        "An HMM is λ = (A, B, π) with hidden states q_t ∈ {1..N}, a_ij = P(q_{t+1}=j | q_t=i), b_j(k) = P(o_t=k | q_t=j), π_i = P(q_1=i). "
        "Assumptions: first-order Markov property and conditional independence of observations given the state. The three questions the algorithms answer: "
        "**evaluation** (Forward/Backward: P(O|λ)), **decoding** (Viterbi: best state path) and **learning** (Baum–Welch: fit λ).")

with tabs[1]:
    st.markdown("### Forward algorithm — evaluation")
    st.markdown("**Input** λ = (A, B, π) and observations O. **Output** α_t(j) and the likelihood P(O|λ).")
    st.latex(r"\alpha_1(j)=\pi_j\,b_j(o_1)\qquad \alpha_t(j)=\Big[\sum_{i}\alpha_{t-1}(i)\,a_{ij}\Big]b_j(o_t)\qquad P(O\mid\lambda)=\sum_j\alpha_T(j)")
    st.code("""# Initialisation
for j in states:   alpha[1][j] = pi[j] * B[j][o1]
# Induction
for t in 2..T:
    for j in states:
        alpha[t][j] = B[j][o_t] * sum_i( alpha[t-1][i] * A[i][j] )
# Termination
P_O = sum_j alpha[T][j]""", language="text")
    st.plotly_chart(G.fig_toy_matrix(toy["alpha"].T, S, tcols, "Forward variables α_t(j)", "Time step (observation)", "Hidden state"), width="stretch")
    a1 = toy["alpha"]
    ex = TOY_B[0, TOY_OBS_SEQ[1]] * float(a1[0] @ TOY_A[:, 0])
    st.markdown(f"**Worked check.** α₂(Sunny) = b_Sunny({obs_names[1]}) × Σᵢ α₁(i)·a_i,Sunny = {TOY_B[0, TOY_OBS_SEQ[1]]:.2f} × {float(a1[0] @ TOY_A[:, 0]):.4f} = **{ex:.4f}** "
                f"(table: {a1[1, 0]:.4f}).")
    st.metric("P(O | λ) from the Forward pass", f"{toy['p_forward']:.6f}")
    ui.callout("plain", "In plain language", f"Given this weather model, the chance of seeing exactly '{' → '.join(obs_names)}' is about {toy['p_forward']:.1%}, adding up every possible hidden weather path.")

with tabs[2]:
    st.markdown("### Backward algorithm — evaluation from the other end")
    st.latex(r"\beta_T(i)=1\qquad \beta_t(i)=\sum_{j}a_{ij}\,b_j(o_{t+1})\,\beta_{t+1}(j)\qquad P(O\mid\lambda)=\sum_i\pi_i\,b_i(o_1)\,\beta_1(i)")
    st.code("""# Initialisation
for i in states:   beta[T][i] = 1
# Induction (backwards)
for t in T-1..1:
    for i in states:
        beta[t][i] = sum_j( A[i][j] * B[j][o_{t+1}] * beta[t+1][j] )
# Termination
P_O = sum_i( pi[i] * B[i][o1] * beta[1][i] )""", language="text")
    st.plotly_chart(G.fig_toy_matrix(toy["beta"].T, S, tcols, "Backward variables β_t(i)", "Time step (observation)", "Hidden state"), width="stretch")
    ident = (toy["alpha"] * toy["beta"]).sum(axis=1)
    c1, c2, c3 = st.columns(3)
    c1.metric("P(O|λ) Forward", f"{toy['p_forward']:.8f}")
    c2.metric("P(O|λ) Backward", f"{toy['p_backward']:.8f}")
    c3.metric("|difference|", f"{abs(toy['p_forward'] - toy['p_backward']):.1e}")
    st.markdown("**Consistency identity:** Σᵢ α_t(i)·β_t(i) = P(O|λ) for **every** t — " + ", ".join(f"t={t + 1}: {v:.6f}" for t, v in enumerate(ident)) + ".")
    ui.callout("plain", "In plain language", "Counting the possibilities forwards or backwards must give the same answer. Having both lets us ask, for any day, 'how likely was each weather state given the whole sequence?' — the building block of learning.")

with tabs[3]:
    st.markdown("### Viterbi algorithm — decoding")
    st.latex(r"\delta_1(j)=\pi_j b_j(o_1)\qquad \delta_t(j)=\max_i\big[\delta_{t-1}(i)a_{ij}\big]b_j(o_t)\qquad \psi_t(j)=\arg\max_i\big[\delta_{t-1}(i)a_{ij}\big]")
    st.code("""# Initialisation
delta[1][j] = pi[j] * B[j][o1];   psi[1][j] = none
# Induction
for t in 2..T, j in states:
    delta[t][j] = B[j][o_t] * max_i( delta[t-1][i] * A[i][j] )
    psi[t][j]   = argmax_i( delta[t-1][i] * A[i][j] )
# Termination
q*_T = argmax_j delta[T][j]
# Backtracking
for t in T-1..1:  q*_t = psi[t+1][ q*_{t+1} ]""", language="text")
    st.plotly_chart(G.fig_toy_trellis(S, obs_names, toy["path_idx"], toy["delta"]), width="stretch")
    c1, c2 = st.columns(2)
    c1.markdown("**Back-pointers ψ** (row t, column state; value = best previous state)")
    c1.dataframe(pd.DataFrame([[S[v] for v in row] for row in toy["psi"]], columns=S, index=tcols), width="stretch")
    c2.metric("Best path", " → ".join(toy["path"]))
    c2.metric("Probability of that path and O", f"{np.exp(toy['logp_viterbi']):.6f}")
    post_arg = [S[i] for i in toy["gamma_initial"].argmax(axis=1)]
    st.markdown(f"For comparison, the individually most probable state at each step (from the Forward–Backward posterior) is {' → '.join(post_arg)}. "
                "Viterbi returns the best **whole path**; posterior decoding returns the best state **per step** — they can differ.")
    ui.callout("plain", "In plain language", "Of all possible weather histories, which single one best explains what we saw? Viterbi finds it without listing them all, by keeping only the best way to reach each state at each step.")

with tabs[4]:
    st.markdown("### Baum–Welch algorithm — learning (EM)")
    st.latex(r"\gamma_t(i)=\frac{\alpha_t(i)\beta_t(i)}{P(O|\lambda)}\qquad \xi_t(i,j)=\frac{\alpha_t(i)a_{ij}b_j(o_{t+1})\beta_{t+1}(j)}{P(O|\lambda)}")
    st.latex(r"\hat\pi_i=\gamma_1(i)\qquad \hat a_{ij}=\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)}\qquad \hat b_j(k)=\frac{\sum_{t:o_t=k}\gamma_t(j)}{\sum_{t}\gamma_t(j)}")
    st.code("""initialise lambda = (A, B, pi)
repeat until |log P(O|lambda_new) - log P(O|lambda_old)| < tol or max_iter:
    E-step: run Forward and Backward; compute gamma_t(i), xi_t(i,j)
    M-step: re-estimate pi, A, B from expected counts
    (states with no expected counts keep their previous rows)""", language="text")
    st.markdown("**E-step posterior γ with the original parameters**")
    st.dataframe(pd.DataFrame(toy["gamma_initial"], columns=S, index=tcols).round(4), width="stretch")
    st.plotly_chart(G.fig_em_likelihood(toy["ll_hist"]), width="stretch")
    mono = all(b >= a - 1e-12 for a, b in zip(toy["ll_hist"], toy["ll_hist"][1:]))
    k1, k2, k3 = st.columns(3)
    k1.metric("Iterations", len(toy["ll_hist"]))
    k2.metric("P(O|λ) start → end", f"{toy['p_hist'][0]:.4f} → {toy['p_hist'][-1]:.4f}")
    k3.metric("Likelihood never decreased", "Yes" if mono else "No")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Learned π**")
    c1.dataframe(pd.DataFrame([toy["pi_learned"]], columns=S, index=["π"]).round(3), width="stretch")
    c2.markdown("**Learned A**")
    c2.dataframe(pd.DataFrame(toy["A_learned"], index=S, columns=S).round(3), width="stretch")
    c3.markdown("**Learned B**")
    c3.dataframe(pd.DataFrame(toy["B_learned"], index=S, columns=O).round(3), width="stretch")
    ui.callout("warn", "Why the learned model looks extreme",
               "With only **three** observations and three symbols, the likelihood is maximised by a deterministic model (Dry→Sunny, Damp→Cloudy, Wet→Rainy, probability → 1). "
               "That is textbook over-fitting and is expected: the example shows the *mechanics* of EM, not a sensible weather model. The crude-oil model has 216 observations and is estimated with restarts.")

with tabs[5]:
    st.markdown("### Numerical stability")
    st.markdown("The textbook recursions multiply many probabilities below 1, which **underflows to 0** for long sequences. Two standard remedies are implemented:")
    st.markdown("- **Scaling** (Rabiner): normalise α_t at every step with c_t = Σⱼ α̃_t(j) and use log P(O|λ) = Σ_t log c_t.\n"
                "- **Log-space** recursions: α̃_t(j) = log b_j(o_t) + logsumexp_i[α̃_{t-1}(i) + log a_ij]. Used for the Gaussian HMM verification and for Baum–Welch here.")
    T = st.select_slider("Sequence length T (simulated from the toy model)", options=[50, 200, 500, 1000, 1500, 3000], value=1500)
    u = _underflow(T)
    c1, c2, c3 = st.columns(3)
    c1.metric("Plain Forward P(O|λ)", f"{u['p_unscaled']:.3e}", "underflow to 0" if u["p_unscaled"] == 0 else "still representable")
    c2.metric("Scaled log P(O|λ)", f"{u['loglik_scaled']:.4f}")
    c3.metric("Log-space log P(O|λ)", f"{u['loglik_log']:.4f}")
    st.caption(f"Scaled and log-space results agree to {abs(u['loglik_scaled'] - u['loglik_log']):.1e}.")
    st.markdown("### Computational complexity")
    st.dataframe(pd.DataFrame({
        "Algorithm": ["Forward", "Backward", "Viterbi", "Baum–Welch (per iteration)", "Gaussian emissions"],
        "Time": ["O(N²T)", "O(N²T)", "O(N²T)", "O(N²T)", "O(T·N·d²)"],
        "Memory": ["O(NT) (O(N) if only the likelihood is needed)", "O(NT)", "O(NT) for back-pointers", "O(NT + N²)", "O(T·N)"],
        "Note": ["N states, T observations", "Same as Forward", "Max replaces sum", "Forward + Backward + re-estimation", "d = number of features (2 here)"],
    }), width="stretch", hide_index=True)
    ui.dual_explanation(
        "Everything stays fast for our sizes: three states and a few hundred months. Working with logarithms keeps the arithmetic reliable even for very long histories.",
        "Interpretation of EM results: Baum–Welch converges to a **local** maximum of the likelihood. Different starting points can reach different optima, which is why the "
        "crude-oil model is fitted from ten random restarts and the best converged log-likelihood is kept (see Regime Detection Pipeline → convergence).")
ui.disclaimer()
