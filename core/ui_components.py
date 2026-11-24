"""CrudeCortex design system: CSS, layout helpers and explanation components.

Rules: no emoji, high contrast (>= 7:1 for body text), minimum 14 px text,
one amber accent, regime colours reserved for regimes. All HTML strings are
single-line so that Markdown never mistakes indentation for a code block.
"""
from __future__ import annotations

import html
import re
from typing import Iterable, Optional

import streamlit as st

from . import config as C
from .config import RunConfig

T = C.THEME

_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {{ --bg:{T['bg']}; --surface:{T['surface']}; --surface2:{T['surface2']}; --border:{T['border']};
  --text:{T['text']}; --muted:{T['muted']}; --accent:{T['accent']}; --gold:{T['gold']};
  --bull:{T['bull']}; --bear:{T['bear']}; --stag:{T['stagnant']}; --info:{T['info']}; }}
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{ font-family:{C.FONT_STACK} !important; }}
.stApp {{ background: var(--bg); color: var(--text); }}
header[data-testid="stHeader"] {{ background: rgba(10,13,16,0.85); backdrop-filter: blur(6px); }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; font-weight: 600; color: var(--text); }}
h1 {{ font-size: 2.1rem; }} h2 {{ font-size: 1.5rem; margin-top: 1.6rem; }} h3 {{ font-size: 1.15rem; }}
p, li {{ font-size: 1rem; line-height: 1.62; }}
a {{ color: var(--accent); }}
code {{ font-family:{C.MONO_STACK} !important; color: var(--gold); background: var(--surface2); border-radius: 4px; padding: 1px 5px; font-size: 0.9em; }}
section[data-testid="stSidebar"] {{ background: var(--surface); border-right: 1px solid var(--border); }}
section[data-testid="stSidebar"] * {{ font-size: 0.95rem; }}
.cc-brand {{ padding: 4px 2px 14px 2px; border-bottom: 1px solid var(--border); margin-bottom: 10px; }}
.cc-brand .name {{ font-size: 1.35rem; font-weight: 600; letter-spacing: 0.02em; color: var(--text); }}
.cc-brand .name b {{ color: var(--accent); font-weight: 600; }}
.cc-brand .tag {{ color: var(--muted); font-size: 0.85rem; margin-top: 2px; }}
.cc-kicker {{ color: var(--accent); font-size: 0.8rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 4px; }}
.cc-title {{ font-size: 2.15rem; font-weight: 600; line-height: 1.15; margin: 0 0 6px 0; }}
.cc-sub {{ color: var(--muted); font-size: 1.08rem; max-width: 62rem; margin-bottom: 18px; }}
.cc-hero {{ background: linear-gradient(180deg, var(--surface) 0%, var(--bg) 100%); border: 1px solid var(--border); border-radius: 14px; padding: 38px 36px 30px 36px; margin-bottom: 22px; position: relative; overflow: hidden; }}
.cc-hero:before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background: var(--accent); }}
.cc-hero .big {{ font-size: 3rem; font-weight: 600; letter-spacing: -0.02em; line-height: 1.05; }}
.cc-hero .big b {{ color: var(--accent); font-weight: 600; }}
.cc-hero .tagline {{ color: var(--gold); font-size: 1.2rem; margin: 8px 0 16px 0; font-style: italic; }}
.cc-hero .lead {{ font-size: 1.1rem; max-width: 52rem; color: var(--text); }}
.cc-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; height: 100%; }}
.cc-card h4 {{ margin: 0 0 6px 0; font-size: 1.05rem; }}
.cc-card p {{ margin: 0; color: var(--muted); font-size: 0.97rem; }}
.cc-kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }}
.cc-kpi .l {{ color: var(--muted); font-size: 0.82rem; letter-spacing: 0.04em; text-transform: uppercase; }}
.cc-kpi .v {{ font-size: 1.7rem; font-weight: 600; margin-top: 2px; font-family:{C.MONO_STACK}; }}
.cc-kpi .s {{ color: var(--muted); font-size: 0.85rem; margin-top: 2px; }}
.cc-kpi.bull .v {{ color: var(--bull); }} .cc-kpi.bear .v {{ color: var(--bear); }} .cc-kpi.stag .v {{ color: var(--stag); }} .cc-kpi.acc .v {{ color: var(--accent); }}
.cc-badge {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 500; border: 1px solid var(--border); background: var(--surface2); color: var(--text); margin-right: 6px; }}
.cc-badge.ok {{ border-color: {T['bull']}; color: {T['bull']}; }} .cc-badge.warn {{ border-color: {T['accent']}; color: {T['accent']}; }}
.cc-badge.err {{ border-color: {T['bear']}; color: {T['bear']}; }} .cc-badge.info {{ border-color: {T['info']}; color: {T['info']}; }}
.cc-callout {{ border-radius: 10px; padding: 14px 18px; margin: 10px 0 14px 0; border: 1px solid var(--border); border-left-width: 4px; background: var(--surface); }}
.cc-callout .t {{ font-weight: 600; margin-bottom: 4px; }}
.cc-callout.plain {{ border-left-color: var(--info); }} .cc-callout.warn {{ border-left-color: var(--accent); }}
.cc-callout.note {{ border-left-color: var(--gold); }} .cc-callout.ok {{ border-left-color: var(--bull); }} .cc-callout.err {{ border-left-color: var(--bear); }}
.cc-callout .t.plain {{ color: var(--info); }} .cc-callout .t.warn {{ color: var(--accent); }} .cc-callout .t.note {{ color: var(--gold); }}
.cc-regime {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }}
.cc-sec {{ margin: 28px 0 6px 0; padding-bottom: 6px; border-bottom: 1px solid var(--border); font-size: 1.3rem; font-weight: 600; }}
.cc-secd {{ color: var(--muted); margin-bottom: 12px; }}
.cc-foot {{ color: var(--muted); font-size: 0.85rem; border-top: 1px solid var(--border); margin-top: 36px; padding-top: 14px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ height: 44px; padding: 0 16px; color: var(--muted); font-size: 0.97rem; }}
.stTabs [aria-selected="true"] {{ color: var(--accent) !important; }}
div[data-testid="stExpander"] {{ border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }}
div[data-testid="stExpander"] summary {{ font-weight: 500; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 8px; border: 1px solid var(--border); background: var(--surface2); color: var(--text); font-weight: 500; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: var(--accent); color: var(--accent); }}
.stButton > button[kind="primary"] {{ background: var(--accent); color: #14100A; border-color: var(--accent); }}
.stButton > button[kind="primary"]:hover {{ background: #F2B556; color: #14100A; }}
div[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
div[data-testid="stMetric"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; }}
@media (max-width: 760px) {{
  .block-container {{ padding-left: 16px; padding-right: 16px; }}
  .cc-hero {{ padding: 24px 18px; }} .cc-hero .big {{ font-size: 2.1rem; }} .cc-title {{ font-size: 1.7rem; }}
}}
"""


def inject_css() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def _inline(md: str) -> str:
    """Tiny inline-Markdown → HTML (bold, code, italics); everything else is escaped."""
    s = html.escape(md)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)
    return s.replace("\n", "<br>")


def page_header(title: str, subtitle: str = "", kicker: str = "") -> None:
    k = f'<div class="cc-kicker">{html.escape(kicker)}</div>' if kicker else ""
    sub = f'<div class="cc-sub">{_inline(subtitle)}</div>' if subtitle else ""
    st.markdown(f'{k}<div class="cc-title">{html.escape(title)}</div>{sub}', unsafe_allow_html=True)


def section(title: str, desc: str = "") -> None:
    d = f'<div class="cc-secd">{_inline(desc)}</div>' if desc else ""
    st.markdown(f'<div class="cc-sec">{html.escape(title)}</div>{d}', unsafe_allow_html=True)


def badge(text: str, tone: str = "") -> str:
    return f'<span class="cc-badge {tone}">{html.escape(text)}</span>'


def badges(items: Iterable[tuple[str, str]]) -> None:
    st.markdown("".join(badge(t, tone) for t, tone in items), unsafe_allow_html=True)


def callout(kind: str, title: str, body: str) -> None:
    """kind: plain | warn | note | ok | err."""
    st.markdown(f'<div class="cc-callout {kind}"><div class="t {kind}">{html.escape(title)}</div>'
                f'<div>{_inline(body)}</div></div>', unsafe_allow_html=True)


def kpis(items: list[tuple[str, str, str, str]]) -> None:
    """items: (label, value, sub, tone) with tone in '', bull, bear, stag, acc."""
    cols = st.columns(len(items))
    for col, (l, v, s, tone) in zip(cols, items):
        col.markdown(f'<div class="cc-kpi {tone}"><div class="l">{html.escape(l)}</div>'
                     f'<div class="v">{html.escape(str(v))}</div><div class="s">{html.escape(s)}</div></div>',
                     unsafe_allow_html=True)


def cards(items: list[tuple[str, str]], per_row: int = 3) -> None:
    for i in range(0, len(items), per_row):
        row = items[i:i + per_row]
        cols = st.columns(per_row)
        for col, (t, b) in zip(cols, row):
            col.markdown(f'<div class="cc-card"><h4>{html.escape(t)}</h4><p>{_inline(b)}</p></div>', unsafe_allow_html=True)


def dual_explanation(plain: str, research: str, plain_title: str = "In plain language",
                     research_title: str = "Research detail — methodology, equations, assumptions") -> None:
    """Plain-language callout (always visible) + collapsible technical section."""
    callout("plain", plain_title, plain)
    with st.expander(research_title):
        st.markdown(research)


def regime_chip(name: str, cmap: Optional[dict] = None) -> str:
    c = (cmap or C.REGIME_COLORS).get(name, T["muted"])
    return f'<span class="cc-regime" style="background:{c}"></span>{html.escape(name)}'


def config_banner(run: RunConfig, extra: str = "") -> None:
    if run.is_research_default:
        st.markdown(badge("Report configuration", "ok") + badge("reproduces submitted GWP2 results", "info")
                    + (badge(extra, "") if extra else ""), unsafe_allow_html=True)
    else:
        st.markdown(badge("Alternative run", "warn") + badge("not the submitted report's results", "warn")
                    + (badge(extra, "") if extra else ""), unsafe_allow_html=True)
        with st.expander(f"Settings that differ from the report ({len(run.deviations())})"):
            for d in run.deviations():
                st.markdown(f"- {d}")


def sidebar_brand(run: RunConfig, window: str = "") -> None:
    with st.sidebar:
        st.markdown(f'<div class="cc-brand"><div class="name">Crude<b>Cortex</b></div>'
                    f'<div class="tag">{html.escape(C.TAGLINE)}</div></div>', unsafe_allow_html=True)
        if run.is_research_default:
            st.markdown(badge("Report configuration", "ok"), unsafe_allow_html=True)
        else:
            st.markdown(badge("Alternative run", "warn"), unsafe_allow_html=True)
        if window:
            st.caption(window)


def sidebar_footer() -> None:
    with st.sidebar:
        st.markdown("---")
        st.caption(f"{C.COURSE}  \n{C.ASSIGNMENT} · {C.GROUP}")
        st.caption("Historical, descriptive research. Not investment advice.")


def disclaimer() -> None:
    st.markdown('<div class="cc-foot">CrudeCortex is an academic research application (MScFE 660, GWP2). '
                'It describes <b>historical</b> regimes in monthly data; it does not forecast prices, '
                'give investment advice or use real-time quotes. Learned dependencies are conditional '
                'associations, not proof of causation.</div>', unsafe_allow_html=True)


def audience_panels(facts: dict[str, str]) -> None:
    """Four audience-specific interpretation panels. ``facts`` supplies computed sentences."""
    tabs = st.tabs(["Individual investors", "Retail market participants", "Energy companies & management",
                    "Quantitative researchers"])
    with tabs[0]:
        callout("plain", "How to read this",
                "Oil prices move in phases. This tool labels each past month as a *Bull*, *Bear* or *Stagnant* "
                "phase using the data alone, and shows how those phases relate to market-wide fear (the VIX) and the economy.")
        st.markdown(
            f"- {facts.get('persistence', '')}\n"
            "- Regimes are labelled with hindsight, using the whole sample, so they describe *what happened*, not what will happen.\n"
            f"- {facts.get('vix', '')}\n"
            "- Use it to build risk awareness: large swings in volatility are a normal feature of oil markets.")
    with tabs[1]:
        callout("plain", "Transitions and volatility",
                "Retail participants often see short-term moves as trends. The model separates *level of return* from *level of volatility*.")
        st.markdown(
            f"- {facts.get('transition', '')}\n"
            "- A high-volatility regime can contain both rallies and sell-offs; volatility is not direction.\n"
            "- Decoded regimes only become certain in hindsight: the Viterbi labels use all months, including later ones.\n"
            "- Limitation: with a few dozen months per regime, estimates are uncertain and may not repeat.")
    with tabs[2]:
        callout("plain", "Scenario analysis and risk monitoring",
                "The historical regimes can serve as *reference scenarios* for stress-testing plans, budgets and hedging policies.")
        st.markdown(
            f"- {facts.get('energy', '')}\n"
            "- Operational planning could compare outcomes under 'Bull-like', 'Bear-like' and 'high-volatility' price paths drawn from history.\n"
            "- Risk monitoring could track whether recent volatility resembles a past high-volatility regime.\n"
            "- This model has **not** been validated for operational forecasting; treat it as a descriptive framework.")
    with tabs[3]:
        callout("note", "Technical outputs",
                "All estimated quantities (π, A, μ, Σ, posteriors), the learned DAG, CPDs and search history are exposed in the HMM and Bayesian pages and exportable as CSV.")
        st.markdown(
            f"- {facts.get('quant', '')}\n"
            "- Check: restart log-likelihoods (local optima), label ordering, discretisation thresholds and edge orientation (score-equivalent classes).\n"
            "- Methodological caveats: Gaussian emissions for fat-tailed returns, 216 monthly observations, structure learned on the same sample as the regimes.")
