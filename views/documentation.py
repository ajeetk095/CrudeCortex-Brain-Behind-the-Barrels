"""Documentation — user guide, glossary, configuration, downloads."""
import json
import os

import streamlit as st

from core import config as C
from core import exports, state, ui_components as ui

run = state.get_run()
ds = state.load_dataset(run)
rh = state.load_hmm(ds, run)
panel, bn = state.load_bn(ds, rh, run)

ui.page_header("Documentation", "How to use the app, key terms, the exact configuration of this session and the original project files.", "Results & documentation")
t = st.tabs(["User guide", "Glossary", "Configuration", "Original documents"])
with t[0]:
    st.markdown(
        "**Suggested route**\n\n"
        "1. *Home → Research Objectives → Data & Indicators* for context and data provenance.\n"
        "2. *HMM Algorithms Lab* for the toy example, then *Market Regime Identification* (rule-based benchmark).\n"
        "3. *HMM Definition & Design*, then *Regime Detection Pipeline* for the fitted model.\n"
        "4. *Bayesian Network Lab* and *Conditional Probability Explorer*.\n"
        "5. *Summary Dashboard* to export everything.\n\n"
        "**Controls.** Change settings in the expanders on Data, Pipeline and Bayesian pages and press **Run analysis**; nothing recomputes until then. "
        "**Reset** restores the research defaults. **Refresh data from FRED** requests live retrieval; if it fails, the app falls back to cache or the bundled snapshot and says so. "
        "A banner shows whenever settings differ from the submitted report.")
with t[1]:
    st.markdown(
        "- **Regime** – a persistent market phase (here Bull, Bear, Stagnant).\n- **HMM** – model with hidden states that emit observable data.\n"
        "- **Transition matrix** – probabilities of moving between regimes month to month.\n- **Baum–Welch** – EM algorithm that learns HMM parameters.\n"
        "- **Viterbi** – finds the single most likely state sequence.\n- **Posterior** – probability of each state given all data.\n"
        "- **Log-likelihood** – how well the model explains the data (higher is better).\n- **BIC/AIC** – fit scores with complexity penalties.\n"
        "- **Bayesian Network** – directed acyclic graph of conditional dependencies.\n- **CPD** – conditional probability table of a node given its parents.\n"
        "- **Hill Climb** – greedy structure search.\n- **VIX** – implied-volatility index of the S&P 500.\n- **ADF** – test for a unit root (non-stationarity).")
with t[2]:
    st.json(exports.config_summary(run, ds, rh, bn))
    st.download_button("Download configuration (JSON)", json.dumps(exports.config_summary(run, ds, rh, bn), indent=2, default=str),
                       "configuration_summary.json", "application/json")
with t[3]:
    st.caption("The original academic documents and notebook are kept unchanged in the repository.")
    d = os.path.join(C.PROJECT_ROOT, "documents")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.lower().endswith(".pdf"):
            with open(os.path.join(d, f), "rb") as fh:
                st.download_button(f, fh.read(), f, "application/pdf", key=f"dl_{f}")
ui.disclaimer()
