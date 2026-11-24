"""Home — what CrudeCortex is, in plain language, with live dataset status."""
import streamlit as st

from core import config as C
from core import state, ui_components as ui
from core.config import REGIME_COLORS

run = state.get_run()
ds = state.load_dataset(run)

st.markdown(
    '<div class="cc-hero"><div class="cc-kicker">MScFE 660 · Risk Management · Group Work Project 2</div>'
    '<div class="big">Crude<b>Cortex</b></div>'
    f'<div class="tagline">{C.TAGLINE}</div>'
    '<div class="lead">A probabilistic research platform that reads the history of WTI crude-oil prices and sorts every month '
    'into one of three market regimes — <b>Bull</b>, <b>Bear</b> or <b>Stagnant</b> — then asks which macroeconomic and '
    'financial indicators are statistically associated with those regimes.</div></div>', unsafe_allow_html=True)

ui.config_banner(run)

ui.kpis([
    ("Monthly panel", f"{ds.master.shape[0]} × {ds.master.shape[1]}", ds.fmt_range("master"), ""),
    ("Modelled months", f"{ds.features.shape[0]} × {ds.features.shape[1]}", ds.fmt_range("features"), "acc"),
    ("ADF statistic", f"{ds.adf['stat']:.3f}", "stationary at 1%" if ds.adf["p"] < 0.01 else "stationarity not established", ""),
    ("Hidden states", str(run.hmm.n_states), "Bull · Bear · Stagnant" if run.hmm.n_states == 3 else "experimental", ""),
])
origins = ds.status["Origin"].unique().tolist()
tone = "ok" if origins == ["Bundled snapshot"] else "warn"
ui.badges([(f"Data: {', '.join(origins)}", tone),
           ("Historical monthly data — not real-time quotes", "info"),
           (f"Latest month in dataset: {ds.full_master.index[-1]:%b %Y}", "")])
for w in ds.warnings:
    st.warning(w)

ui.section("What does this platform do?", "No finance background needed.")
ui.cards([
    ("The question", "Oil does not move in one steady way. Some stretches are calm rallies, some are violent sell-offs, "
                     "others drift. Can we identify those stretches objectively from the data?"),
    ("The method", "A **Hidden Markov Model** treats the market's phase as an unseen state that changes over time and "
                   "reveals itself through returns and volatility. A **Bayesian Network** then maps how the phase relates to "
                   "market fear (VIX), industrial activity, energy inflation and the dollar."),
    ("The caution", "Everything here is **descriptive and historical**. Regimes are labelled in hindsight; dependencies are "
                    "associations, not causes. Nothing on this site is a forecast or investment advice."),
], per_row=3)

ui.section("The three regimes", "How the model's states are named")
c1, c2, c3 = st.columns(3)
for col, (name, text) in zip((c1, c2, c3), [
    ("Bull", "The state with the **highest** average monthly return. Typically calmer, steadier markets."),
    ("Stagnant", "The **middle** state by average return. The name reflects that ordering, not calm: in the submitted report this state is in fact the most volatile."),
    ("Bear", "The state with the **lowest** average return: sustained declines."),
]):
    col.markdown(f'<div class="cc-card"><h4>{ui.regime_chip(name)}</h4><p>{text}</p></div>', unsafe_allow_html=True)
st.caption("Labels are assigned by ranking each state's mean return (see HMM Definition & Design). "
           "The rule-based benchmark on the Market Regime Identification page is a separate, simpler classification.")

ui.section("Two models, one pipeline")
a, b = st.columns(2)
with a:
    st.markdown("#### Hidden Markov Model")
    st.markdown("Three hidden states, two observed features (monthly log-return and 12-month annualised volatility), "
                "Gaussian emissions with full covariance, estimated by Baum–Welch (EM) with ten random restarts and decoded with the Viterbi algorithm.")
    st.page_link("views/regime_pipeline.py", label="Open the Regime Detection Pipeline", icon=":material/timeline:")
with b:
    st.markdown("#### Bayesian Network")
    st.markdown("Five discrete variables — the WTI regime plus four macro-financial indicators — whose dependency structure is learned by "
                "Hill Climb Search with the BIC score, followed by conditional-probability tables and exact inference.")
    st.page_link("views/bayesian_network_lab.py", label="Open the Bayesian Network Lab", icon=":material/hub:")

ui.section("Explore the research")
n1, n2, n3, n4 = st.columns(4)
n1.page_link("views/data_indicators.py", label="Data & Indicators", icon=":material/database:")
n2.page_link("views/hmm_algorithms_lab.py", label="HMM Algorithms Lab", icon=":material/functions:")
n3.page_link("views/summary_dashboard.py", label="Summary Dashboard", icon=":material/dashboard:")
n4.page_link("views/methodology.py", label="Methodology & Limitations", icon=":material/rule:")

ui.section("Academic project")
st.markdown(f"**{C.COURSE}** · {C.ASSIGNMENT} · {C.GROUP}  \n"
            f"**Research topic:** {C.TOPIC}  \n"
            f"**Submitted report period:** January 2005 – December 2024 source window "
            f"(effective modelling sample {ds.fmt_range('features') if ds.is_report_window else 'Jan 2007 – Dec 2024'}; see Data & Indicators for why).  \n"
            f"**Developed by** [Ajeet Kumar]({C.DEVELOPER['github']}) · "
            f"[Source code]({C.GITHUB_REPO})")
