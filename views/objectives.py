"""Research Objectives — aim, scope, questions and where each official GWP2 step lives."""
import streamlit as st

from core import config as C
from core import ui_components as ui

ui.page_header("Research Objectives", "What the project sets out to do, what it deliberately does not do, and where every official step is covered.",
               "Research overview")

ui.section("Objective")
st.markdown("Develop a **probabilistic framework** for identifying crude-oil market regimes and for examining their "
            "**conditional relationships** with macroeconomic and financial indicators, using Hidden Markov Models "
            "and Bayesian Networks on monthly WTI data.")
ui.dual_explanation(
    "We want an objective, data-driven way to say *which kind of market a given month belonged to* — and to see whether "
    "things like market fear or the strength of the dollar tend to go along with each kind of market.",
    "**Research questions**\n\n1. Can a three-state Gaussian HMM on (log-return, 12M annualised volatility) recover economically interpretable WTI regimes?\n"
    "2. How persistent are the regimes (transition matrix, spell lengths)?\n"
    "3. How does the HMM labelling compare with a transparent rule-based benchmark?\n"
    "4. Which macro-financial variables are conditionally associated with the regime in a score-learned Bayesian Network (Hill Climb, BIC)?\n\n"
    "**Scope** — GWP2 is a *methodology and model-development* project. The platform does not optimise portfolios, "
    "does not claim to forecast oil prices and does not replace the HMM / Bayesian-network methodology with other models.")

ui.section("In scope and out of scope")
a, b = st.columns(2)
with a:
    st.markdown("**In scope**")
    st.markdown("- Forward, Backward, Viterbi and Baum–Welch algorithms on a discrete toy HMM\n"
                "- Rule-based Bull / Bear / Stagnant benchmark\n- Three-state Gaussian HMM on WTI\n"
                "- Bayesian Network structure learning, CPDs and conditional inference\n- Transparent data pipeline, limitations and exports")
with b:
    st.markdown("**Out of scope**")
    st.markdown("- Price forecasting or trading signals\n- Portfolio construction\n- Machine-learning models unrelated to HMM / BN\n"
                "- Real-time market data or intraday analysis\n- Causal claims")

ui.section("Where each official GWP2 step lives")
rows = [
    ("1", "Forward, Backward, Viterbi, Baum–Welch: pseudocode and toy example", "views/hmm_algorithms_lab.py", "HMM Algorithms Lab"),
    ("2", "Bull / Bear / Stagnant identification and illustration", "views/regime_identification.py", "Market Regime Identification"),
    ("3", "Team HMM definition (updated reference)", "views/hmm_definition.py", "HMM Definition & Design"),
    ("4", "Research design: macro research, regime process, BN training", "views/hmm_definition.py", "HMM Definition & Design"),
    ("6", "Transformation, parameter learning, Viterbi decoding", "views/regime_pipeline.py", "Regime Detection Pipeline"),
    ("7", "Latent meaning of hidden states", "views/regime_pipeline.py", "Regime Detection Pipeline"),
    ("8", "Hill Climb Search theory", "views/bayesian_network_lab.py", "Bayesian Network Lab"),
    ("9", "Working example, CPDs, inference", "views/bayesian_network_lab.py", "Bayesian Network Lab"),
]
for step, task, path, label in rows:
    c1, c2, c3 = st.columns([0.6, 5, 3])
    c1.markdown(f"**Step {step}**")
    c2.markdown(task)
    c3.page_link(path, label=label)
st.caption("Step 5 of the official brief is a planning step and has no separate software output.")
ui.disclaimer()
