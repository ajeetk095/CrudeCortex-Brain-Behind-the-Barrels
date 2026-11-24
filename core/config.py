"""Central configuration for CrudeCortex GWP2.

Holds (a) static metadata for the five source series, (b) the *research
defaults* that reproduce the submitted MScFE 660 GWP2 report, (c) immutable
run-configuration dataclasses that double as cache keys, and (d) the design
tokens shared by charts and UI.

Nothing here is a computed result. ``REPORT_REFERENCE`` lists values *printed
in the submitted report / executed notebook*; they are used only to label
comparisons and by the validation tests, never to populate a result display.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Literal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_DIR = os.path.join(PROJECT_ROOT, "data", "fred")
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(PROJECT_ROOT, "data", "cache"))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
HTTP_TIMEOUT_S = float(os.environ.get("HTTP_TIMEOUT_S", "15"))
HTTP_RETRIES = int(os.environ.get("HTTP_RETRIES", "3"))

# Source window used by the report/notebook (`START, END`). The *effective*
# window is shorter because DTWEXBGS only begins in January 2006.
START = "2005-01-01"
END = "2024-12-31"
REPORT_FIRST_FEATURE_MONTH = "2007-01-01"
REPORT_LAST_MONTH = "2024-12-01"
ROLLING_WINDOW = 12            # months in the rolling volatility window
WARMUP_MONTHS = ROLLING_WINDOW  # 1 for the first return + 11 to fill the window

PROJECT_NAME = "CrudeCortex"
TAGLINE = "The Brain Behind the Barrel."
COURSE = "MScFE 660 — Risk Management"
ASSIGNMENT = "Group Work Project 2 (GWP2)"
GROUP = "Group 14660"
TOPIC = "Hidden Markov Models and Bayesian Networks for Crude Oil Price Regime Detection"
GITHUB_REPO = "https://github.com/ajeetk095/CrudeCortex-GWP2"
LIVE_APP = "https://crudecortex295.streamlit.app/"

# --------------------------------------------------------------------------
# Source series metadata
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesSpec:
    key: str            # column name in the master panel
    series_id: str      # provider identifier
    label: str
    description: str
    provider: str
    native_frequency: str
    units: str
    role: str
    monthly_rule: str   # how the native series is turned into a monthly value


SERIES: dict[str, SeriesSpec] = {
    "WTI": SeriesSpec(
        "WTI", "WTISPLC", "WTI crude oil spot price",
        "Cushing, OK WTI spot price FOB, monthly average", "FRED (EIA)",
        "Monthly", "USD per barrel", "Target variable",
        "Native monthly average (no resampling)"),
    "IndPro": SeriesSpec(
        "IndPro", "INDPRO", "US industrial production",
        "Industrial Production Index (2017=100), seasonally adjusted", "FRED (Federal Reserve Board)",
        "Monthly", "Index, 2017=100", "Demand proxy",
        "Native monthly (no resampling)"),
    "CPI_Nrg": SeriesSpec(
        "CPI_Nrg", "CPIENGSL", "US CPI: energy",
        "CPI for All Urban Consumers: Energy, seasonally adjusted", "FRED (BLS)",
        "Monthly", "Index, 1982-84=100", "Energy-inflation measure",
        "Native monthly (no resampling)"),
    "DXY": SeriesSpec(
        "DXY", "DTWEXBGS", "Trade-weighted US dollar index (broad)",
        "Nominal Broad US Dollar Index; begins January 2006", "FRED (Federal Reserve Board)",
        "Daily", "Index, Jan 2006=100", "Dollar-strength proxy",
        "Last daily observation of each month (as in the notebook)"),
    "VIX": SeriesSpec(
        "VIX", "VIXCLS", "CBOE Volatility Index (VIX)",
        "CBOE VIX daily close", "FRED (CBOE) · fallback Yahoo Finance ^VIX",
        "Daily", "Index points", "Risk-aversion proxy",
        "Report: month-end close. Alternative: monthly mean of daily closes"),
}

MACRO_KEYS = ("VIX", "IndPro", "CPI_Nrg", "DXY")
SERIES_BY_ID = {s.series_id: s for s in SERIES.values()}

# --------------------------------------------------------------------------
# Regimes and BN node names
# --------------------------------------------------------------------------
REGIME_ORDER_3 = ["Bear", "Stagnant", "Bull"]        # ascending mean return
BN_LEVEL_NODES = {"VIX": "VIX_Level", "IndPro": "IndPro_Level",
                  "CPI_Nrg": "CPI_Level", "DXY": "DXY_Level"}
BN_TREND_NODES = {"VIX": "VIX_Level", "IndPro": "IndPro_Trend",
                  "CPI_Nrg": "CPI_Nrg_Trend", "DXY": "DXY_Trend"}
TARGET_NODE = "WTI_Regime"

# --------------------------------------------------------------------------
# Reference values printed in the submitted report / executed notebook
# --------------------------------------------------------------------------
REPORT_REFERENCE = {
    "source": "MScFE 660 GWP2 report (Group 14660) and executed notebook",
    "panel_shape": (228, 5),
    "feature_shape": (216, 2),
    "adf_statistic": -9.7807,
    "hmm_loglik": -305.9770,
    "hmm_regime_counts": {"Bull": 129, "Bear": 59, "Stagnant": 28},
    "hmm_mean_vix": {"Bear": 18.7, "Bull": 17.9, "Stagnant": 32.8},
    "hmm_mean_return_pct": {"Bear": -1.49, "Bull": 0.84, "Stagnant": -0.31},
    "hmm_mean_vol_pct": {"Bear": 35.1, "Bull": 21.3, "Stagnant": 77.0},
    "rule_based_counts": {"Bull": 79, "Bear": 50, "Stagnant": 87},
    "bn_edges_report": [("VIX_Level", "WTI_Regime")],
    # Report Table 14: P(Bear), P(Stagnant), P(Bull) by VIX level
    "table14": {
        "High": {"Bear": 0.225, "Stagnant": 0.352, "Bull": 0.423},
        "Low": {"Bear": 0.296, "Stagnant": 0.000, "Bull": 0.704},
        "Medium": {"Bear": 0.297, "Stagnant": 0.041, "Bull": 0.662},
    },
}


# --------------------------------------------------------------------------
# Run configuration (immutable => hashable => safe cache keys)
# --------------------------------------------------------------------------
VixAggregation = Literal["month_end_close", "monthly_mean"]
DataMode = Literal["snapshot", "live"]
Discretisation = Literal["report", "tertile"]
BNVariables = Literal["level", "trend"]
BNScore = Literal["bic", "aic"]
CovType = Literal["full", "diag", "tied", "spherical"]


@dataclass(frozen=True)
class DataConfig:
    """Which data enter the analysis. Defaults reproduce the report."""
    feature_start: str = REPORT_FIRST_FEATURE_MONTH      # first modelled month
    feature_end: str = REPORT_LAST_MONTH                 # last modelled month
    vix_aggregation: VixAggregation = "month_end_close"
    data_mode: DataMode = "snapshot"
    refresh_nonce: int = 0   # bumped by the "Refresh data" button to defeat caches


@dataclass(frozen=True)
class HMMConfig:
    """Gaussian HMM settings. Defaults = submitted report (Table 12)."""
    n_states: int = 3
    n_restarts: int = 10
    n_iter: int = 200
    tol: float = 1e-4
    covariance_type: CovType = "full"


@dataclass(frozen=True)
class BNConfig:
    """Bayesian-network settings. Defaults = report-faithful configuration."""
    max_parents: int = 2
    score: BNScore = "bic"
    discretisation: Discretisation = "report"
    variables: BNVariables = "trend"      # notebook definition (reproduces the report's single edge)
    indicators: tuple[str, ...] = MACRO_KEYS


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    bn: BNConfig = field(default_factory=BNConfig)

    def as_dict(self) -> dict:
        return asdict(self)

    def deviations(self) -> list[str]:
        """Human-readable list of settings that differ from the research defaults."""
        out: list[str] = []
        d, h, b = self.data, self.hmm, self.bn
        dd, hd, bd = DataConfig(), HMMConfig(), BNConfig()
        for name, a, b_ in [
            ("Feature start", d.feature_start, dd.feature_start),
            ("Feature end", d.feature_end, dd.feature_end),
            ("VIX aggregation", d.vix_aggregation, dd.vix_aggregation),
            ("Data source", d.data_mode, dd.data_mode),
            ("HMM states", h.n_states, hd.n_states),
            ("HMM restarts", h.n_restarts, hd.n_restarts),
            ("HMM max iterations", h.n_iter, hd.n_iter),
            ("HMM tolerance", h.tol, hd.tol),
            ("HMM covariance", h.covariance_type, hd.covariance_type),
            ("BN max parents", b.max_parents, bd.max_parents),
            ("BN score", b.score, bd.score),
            ("BN discretisation", b.discretisation, bd.discretisation),
            ("BN variable definition", b.variables, bd.variables),
            ("BN indicators", b.indicators, bd.indicators),
        ]:
            if a != b_:
                out.append(f"{name}: {a} (report: {b_})")
        return out

    @property
    def is_research_default(self) -> bool:
        return not self.deviations()


DEFAULT_RUN = RunConfig()

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
THEME = {
    "bg": "#0A0D10",          # deep petroleum black
    "surface": "#11161B",     # dark charcoal
    "surface2": "#182028",
    "border": "#27313A",
    "text": "#ECEFF1",
    "muted": "#9AA7B1",
    "accent": "#E8A33D",      # refined amber
    "accent_dim": "#B07A22",
    "gold": "#C9A45C",        # oil-gold
    "bull": "#5DB37E",
    "bear": "#D4574C",
    "stagnant": "#C8B26A",
    "info": "#6FA8DC",
}
REGIME_COLORS = {"Bull": THEME["bull"], "Bear": THEME["bear"], "Stagnant": THEME["stagnant"]}
FONT_STACK = ("'IBM Plex Sans', 'Inter', -apple-system, 'Segoe UI', Roboto, "
              "'Helvetica Neue', Arial, sans-serif")
MONO_STACK = "'IBM Plex Mono', 'SFMono-Regular', Menlo, Consolas, monospace"

DEVELOPER = {
    "name": "Ajeet Kumar",
    "positioning": "Quantitative Finance · Financial Engineering · Artificial Intelligence · Data Science",
    "github": "https://github.com/ajeetk095",
    "linkedin": "https://www.linkedin.com/in/ajeek095/",
    "photo_candidates": ["ajeet_kumar.jpg", "ajeet_kumar.jpeg", "ajeet_kumar.png", "ajeet_kumar.webp"],
}
