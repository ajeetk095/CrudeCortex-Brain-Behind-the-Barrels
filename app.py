"""CrudeCortex — GWP2 · entry point.

Run locally:  streamlit run app.py
Defines the page set, injects the design system and renders the shared sidebar.
All research logic lives in ``core/``; each page is a thin view in ``views/``.
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import config as C                      # noqa: E402
from core import state, ui_components as ui       # noqa: E402
from core.data_sources import DataError           # noqa: E402

st.set_page_config(page_title="CrudeCortex — GWP2", page_icon=":material/oil_barrel:", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

pages = {
    "Research Overview": [
        st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
        st.Page("views/objectives.py", title="Research Objectives", icon=":material/flag:"),
        st.Page("views/data_indicators.py", title="Data & Indicators", icon=":material/database:"),
    ],
    "HMM Research": [
        st.Page("views/hmm_algorithms_lab.py", title="HMM Algorithms Lab", icon=":material/functions:"),
        st.Page("views/regime_identification.py", title="Market Regime Identification", icon=":material/candlestick_chart:"),
        st.Page("views/hmm_definition.py", title="HMM Definition & Design", icon=":material/schema:"),
        st.Page("views/regime_pipeline.py", title="Regime Detection Pipeline", icon=":material/timeline:"),
    ],
    "Bayesian Research": [
        st.Page("views/bayesian_network_lab.py", title="Bayesian Network Lab", icon=":material/hub:"),
        st.Page("views/conditional_explorer.py", title="Conditional Probability Explorer", icon=":material/tune:"),
    ],
    "Results & Documentation": [
        st.Page("views/summary_dashboard.py", title="Summary Dashboard", icon=":material/dashboard:"),
        st.Page("views/methodology.py", title="Methodology & Limitations", icon=":material/rule:"),
        st.Page("views/documentation.py", title="Documentation", icon=":material/menu_book:"),
        st.Page("views/developer_profile.py", title="Developer Profile", icon=":material/person:"),
    ],
}

run = state.get_run()
window = ""
try:
    _ds = state._cached_dataset(state._j(run.data))
    window = f"Modelled months {_ds.fmt_range('features')} · {len(_ds.features)} obs"
except DataError:
    window = "Data unavailable — see Data & Indicators"
ui.sidebar_brand(run, window)
nav = st.navigation(pages)
ui.sidebar_footer()
nav.run()
