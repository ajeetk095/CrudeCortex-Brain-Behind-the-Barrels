"""Plotly figure builders for CrudeCortex (all charts share one dark template).

Every builder returns a ``plotly.graph_objects.Figure`` with a descriptive
title, labelled axes with units, hover text and the shared regime colours.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config as C
from .bn_engine import BNModel, cpd_frame
from .hmm_engine import RegimeHMM, compute_spells

T = C.THEME
FONT = dict(family=C.FONT_STACK, color=T["text"], size=13)
LEVEL_COLORS = {"Low": "#6C7A89", "Medium": "#B58A3A", "High": "#E8A33D",
                "Down": "#6C7A89", "Flat": "#B58A3A", "Up": "#E8A33D",
                "Falling": "#6C7A89", "Rising": "#E8A33D"}


def color_map(names: Sequence[str]) -> dict[str, str]:
    """Consistent colours: Bear/Stagnant/Bull fixed; other state counts get a red→gold→green ramp."""
    if all(n in C.REGIME_COLORS for n in names):
        return {n: C.REGIME_COLORS[n] for n in names}
    ramp = ["#D4574C", "#E08A4B", "#C8B26A", "#9CBF72", "#5DB37E"]
    n = len(names)
    idx = np.linspace(0, len(ramp) - 1, n)
    return {nm: ramp[int(round(i))] for nm, i in zip(names, idx)}


def _hex_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _layout(fig: go.Figure, title: str, xtitle: Optional[str] = None, ytitle: Optional[str] = None,
            height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16, color=T["text"])),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=T["surface"],
        font=FONT, height=height, margin=dict(l=60, r=24, t=64, b=56),
        showlegend=legend, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(bgcolor=T["surface2"], bordercolor=T["border"], font=dict(color=T["text"])),
    )
    fig.update_xaxes(gridcolor=T["border"], zerolinecolor=T["border"], linecolor=T["border"],
                     title=dict(text=xtitle) if xtitle else None)
    fig.update_yaxes(gridcolor=T["border"], zerolinecolor=T["border"], linecolor=T["border"],
                     title=dict(text=ytitle) if ytitle else None)
    return fig


def _shade(fig: go.Figure, labels: pd.Series, cmap: dict, row=None, col=None, opacity: float = 0.16):
    spells = compute_spells(labels)
    for _, s in spells.iterrows():
        kw = dict(x0=s["Start"], x1=s["End"] + pd.offsets.MonthBegin(1), fillcolor=cmap.get(s["Regime"], "#888"),
                  opacity=opacity, line_width=0, layer="below")
        if row:
            fig.add_vrect(row=row, col=col or 1, **kw)
        else:
            fig.add_vrect(**kw)


def _legend_proxies(fig: go.Figure, cmap: dict):
    for n, c in cmap.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=n,
                                 marker=dict(size=10, color=c, symbol="square"), hoverinfo="skip"))


# ---------------------------------------------------------------- market data

def fig_price(master: pd.DataFrame, labels: Optional[pd.Series] = None, names=None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=master.index, y=master["WTI"], mode="lines", name="WTI spot (monthly avg)",
                               line=dict(color=T["accent"], width=2),
                               hovertemplate="%{x|%b %Y}<br>WTI: $%{y:,.2f}/bbl<extra></extra>"))
    if labels is not None:
        cmap = color_map(names or list(labels.unique()))
        _shade(fig, labels, cmap)
        _legend_proxies(fig, cmap)
    fig.update_xaxes(tickformat="%Y")
    return _layout(fig, "WTI crude oil — monthly average spot price", "Month", "USD per barrel")


def fig_returns(features: pd.DataFrame, labels: Optional[pd.Series] = None, names=None) -> go.Figure:
    r = features["WTI_ret"] * 100
    fig = go.Figure(go.Bar(x=features.index, y=r, name="Monthly log-return",
                           marker_color=np.where(r >= 0, T["bull"], T["bear"]),
                           hovertemplate="%{x|%b %Y}<br>log-return: %{y:.2f}%<extra></extra>"))
    if labels is not None:
        cmap = color_map(names or list(labels.unique()))
        _shade(fig, labels, cmap, opacity=0.12)
    fig.update_xaxes(tickformat="%Y")
    return _layout(fig, "WTI monthly log-return", "Month", "Log-return (%)", legend=False)


def fig_vol(features: pd.DataFrame, labels: Optional[pd.Series] = None, names=None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=features.index, y=features["WTI_vol12"] * 100, mode="lines",
                               name="12M rolling volatility", line=dict(color=T["info"], width=2),
                               hovertemplate="%{x|%b %Y}<br>vol: %{y:.1f}% p.a.<extra></extra>"))
    if labels is not None:
        cmap = color_map(names or list(labels.unique()))
        _shade(fig, labels, cmap, opacity=0.12)
    fig.update_xaxes(tickformat="%Y")
    return _layout(fig, "WTI 12-month rolling volatility (annualised)", "Month", "Annualised volatility (%)", legend=False)


# ----------------------------------------------------------------- regimes

def fig_regime_timeline(labels: pd.Series, names: Sequence[str], title: str) -> go.Figure:
    """Swim-lane timeline: one row per regime, so colour is never the only cue."""
    cmap = color_map(names)
    spells = compute_spells(labels)
    fig = go.Figure()
    for n in names:
        sp = spells[spells["Regime"] == n]
        if sp.empty:
            continue
        ends = sp["End"] + pd.offsets.MonthBegin(1)
        widths = (ends - sp["Start"]).dt.total_seconds() * 1000
        fig.add_trace(go.Bar(
            x=widths, y=[n] * len(sp), base=sp["Start"], orientation="h", name=n,
            marker=dict(color=cmap[n], line=dict(width=0)),
            customdata=np.stack([sp["Start"].dt.strftime("%b %Y"), sp["End"].dt.strftime("%b %Y"), sp["Months"]], axis=1),
            hovertemplate=f"<b>{n}</b><br>%{{customdata[0]}} – %{{customdata[1]}}<br>%{{customdata[2]}} months<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=list(names)[::-1])
    fig.update_xaxes(type="date", tickformat="%Y")
    fig.update_layout(barmode="overlay", bargap=0.25)
    return _layout(fig, title, "Month", "Regime", height=300, legend=False)


def fig_regime_distribution(series: dict[str, pd.Series], names: Sequence[str]) -> go.Figure:
    """Grouped bars: months per regime for each labelling (e.g. HMM vs rule-based)."""
    cmap = color_map(names)
    fig = go.Figure()
    for i, (label, s) in enumerate(series.items()):
        counts = s.value_counts().reindex(names).fillna(0)
        fig.add_trace(go.Bar(
            x=list(names), y=counts.values, name=label,
            marker=dict(color=[cmap[n] for n in names], opacity=1.0 if i == 0 else 0.55,
                        line=dict(color=T["text"] if i else "rgba(0,0,0,0)", width=1)),
            text=[f"{int(v)}" for v in counts.values], textposition="outside",
            hovertemplate=f"{label}<br>%{{x}}: %{{y}} months<extra></extra>"))
    fig.update_layout(barmode="group")
    return _layout(fig, "Regime distribution (months)", "Regime", "Months")


def fig_duration(spells: pd.DataFrame, names: Sequence[str]) -> go.Figure:
    cmap = color_map(names)
    fig = go.Figure()
    for n in names:
        sp = spells[spells["Regime"] == n]
        fig.add_trace(go.Box(y=sp["Months"], name=n, marker_color=cmap[n], boxpoints="all", jitter=0.4,
                             pointpos=0, line=dict(color=cmap[n]), fillcolor=_hex_rgba(cmap[n], 0.18),
                             hovertemplate=f"{n}: %{{y}} months<extra></extra>"))
    return _layout(fig, "Regime spell lengths (consecutive months in one regime)", "Regime", "Spell length (months)", legend=False)


def fig_transition_heatmap(A: np.ndarray, names: Sequence[str], title="Transition probability matrix P(next | current)") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=A, x=[f"to {n}" for n in names], y=[f"from {n}" for n in names], zmin=0, zmax=1,
        colorscale=[[0, T["surface"]], [0.5, T["accent_dim"]], [1, T["accent"]]],
        text=[[f"{v:.3f}" for v in row] for row in A], texttemplate="%{text}", textfont=dict(size=14),
        hovertemplate="%{y} → %{x}<br>probability %{z:.4f}<extra></extra>", colorbar=dict(title="Prob.")))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, "Next month's regime", "Current regime", height=360, legend=False)


def fig_state_means(rh: RegimeHMM) -> go.Figure:
    cmap = color_map(rh.state_names)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Mean monthly log-return (%)", "Mean annualised volatility (%)"))
    cols = [cmap[n] for n in rh.state_names]
    fig.add_trace(go.Bar(x=rh.state_names, y=rh.means_orig[:, 0] * 100, marker_color=cols,
                         hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>"), 1, 1)
    fig.add_trace(go.Bar(x=rh.state_names, y=rh.means_orig[:, 1] * 100, marker_color=cols,
                         hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>"), 1, 2)
    _layout(fig, "Model-estimated state means (emission parameters, original units)", height=340, legend=False)
    fig.update_yaxes(title_text="Percent", row=1, col=1)
    fig.update_yaxes(title_text="Percent p.a.", row=1, col=2)
    return fig


def fig_state_covariances(rh: RegimeHMM) -> go.Figure:
    n = len(rh.state_names)
    fig = make_subplots(rows=1, cols=n, subplot_titles=rh.state_names, horizontal_spacing=0.06)
    lab = ["Return", "Volatility"]
    for i in range(n):
        S = rh.covars_std[i]
        fig.add_trace(go.Heatmap(z=S, x=lab, y=lab, colorscale="RdBu_r", zmid=0, showscale=(i == n - 1),
                                 text=[[f"{v:.3f}" for v in r] for r in S], texttemplate="%{text}",
                                 hovertemplate="cov(%{y},%{x}) = %{z:.4f}<extra></extra>"), 1, i + 1)
        fig.update_yaxes(autorange="reversed", row=1, col=i + 1)
    return _layout(fig, "Emission covariance matrices (standardised units)", height=330, legend=False)


def _ellipse(mean, cov, k=1.0, n=80):
    vals, vecs = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, n)
    circ = np.stack([np.cos(t), np.sin(t)])
    pts = (vecs @ np.diag(k * np.sqrt(np.maximum(vals, 0))) @ circ).T + mean
    return pts


def fig_feature_space(features: pd.DataFrame, rh: RegimeHMM) -> go.Figure:
    """Observations coloured by decoded regime with ±1σ Gaussian ellipses (original units)."""
    cmap = color_map(rh.state_names)
    fig = go.Figure()
    for i, n in enumerate(rh.state_names):
        m = rh.labels == n
        fig.add_trace(go.Scatter(x=features.loc[m, "WTI_ret"] * 100, y=features.loc[m, "WTI_vol12"] * 100,
                                 mode="markers", name=n, marker=dict(color=cmap[n], size=7, opacity=0.8),
                                 text=features.index[m].strftime("%b %Y"),
                                 hovertemplate="%{text}<br>return %{x:.2f}%<br>vol %{y:.1f}%<extra>" + n + "</extra>"))
        e = _ellipse(rh.means_orig[i], rh.covars_orig[i]) * 100
        fig.add_trace(go.Scatter(x=e[:, 0], y=e[:, 1], mode="lines", line=dict(color=cmap[n], width=2, dash="dot"),
                                 showlegend=False, hoverinfo="skip"))
    return _layout(fig, "Observations and ±1σ emission ellipses", "Monthly log-return (%)", "Annualised volatility (%)", height=420)


def fig_viterbi_posterior(rh: RegimeHMM) -> go.Figure:
    cmap = color_map(rh.state_names)
    fig = go.Figure()
    for n in rh.state_names:
        fig.add_trace(go.Scatter(x=rh.posteriors.index, y=rh.posteriors[n], mode="lines", stackgroup="p", name=n,
                                 line=dict(width=0.5, color=cmap[n]), fillcolor=_hex_rgba(cmap[n], 0.75),
                                 hovertemplate="%{x|%b %Y}<br>" + n + ": %{y:.3f}<extra></extra>"))
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(tickformat="%Y")
    return _layout(fig, "Smoothed state probabilities P(state | all data)", "Month", "Probability")


def fig_convergence(rh: RegimeHMM) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        subplot_titles=("EM progress of the selected run", "Final log-likelihood by restart"))
    h = rh.convergence
    fig.add_trace(go.Scatter(x=list(range(1, len(h) + 1)), y=h, mode="lines+markers", name="log-likelihood",
                             line=dict(color=T["accent"]), hovertemplate="iteration %{x}<br>%{y:.3f}<extra></extra>"), 1, 1)
    r = rh.restarts.dropna(subset=["Log-likelihood"])
    fig.add_trace(go.Bar(x=r["Seed"].astype(str), y=r["Log-likelihood"],
                         marker_color=[T["accent"] if s else T["border"] for s in r["Selected"]],
                         hovertemplate="seed %{x}<br>%{y:.3f}<extra></extra>"), 1, 2)
    lo = float(r["Log-likelihood"].min())
    hi = float(r["Log-likelihood"].max())
    fig.update_yaxes(range=[lo - 0.05 * abs(hi - lo) - 1, hi + 0.05 * abs(hi - lo) + 1], row=1, col=2)
    fig.update_xaxes(title_text="EM iteration", row=1, col=1)
    fig.update_xaxes(title_text="Random seed", row=1, col=2)
    fig.update_yaxes(title_text="Log-likelihood", row=1, col=1)
    fig.update_yaxes(title_text="Log-likelihood", row=1, col=2)
    return _layout(fig, "HMM convergence diagnostics", height=360, legend=False)


# ------------------------------------------------------------------ macro

def fig_macro(master: pd.DataFrame, labels: Optional[pd.Series] = None, names=None, indexed: bool = True) -> go.Figure:
    keys = [k for k in ("WTI", "VIX", "IndPro", "CPI_Nrg", "DXY") if k in master]
    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=[C.SERIES[k].label for k in keys])
    cmap = color_map(names or list(labels.unique())) if labels is not None else {}
    pal = [T["accent"], T["info"], "#9B8AC4", T["gold"], "#7FB7A4"]
    for i, k in enumerate(keys, 1):
        y = master[k] / master[k].iloc[0] * 100 if indexed and k != "VIX" else master[k]
        fig.add_trace(go.Scatter(x=master.index, y=y, mode="lines", line=dict(color=pal[i - 1], width=1.8),
                                 name=k, hovertemplate="%{x|%b %Y}<br>" + C.SERIES[k].label + ": %{y:,.2f}<extra></extra>"), i, 1)
        unit = "index points" if k == "VIX" else ("USD/bbl" if k == "WTI" and not indexed else "start = 100")
        fig.update_yaxes(title_text=unit, row=i, col=1, title_font=dict(size=11))
        if labels is not None:
            _shade(fig, labels, cmap, row=i, opacity=0.10)
    fig.update_xaxes(tickformat="%Y")
    _layout(fig, "Macro and financial indicators (monthly) with HMM regime shading", height=150 * len(keys) + 90, legend=False)
    fig.update_xaxes(title_text="Month", row=len(keys), col=1)
    return fig


# ------------------------------------------------------------ bayes net

def _positions(nodes: list[str], target: str) -> dict[str, tuple[float, float]]:
    pos = {}
    others = [n for n in nodes if n != target]
    if target in nodes:
        pos[target] = (0.0, 0.0)
    for i, n in enumerate(others):
        a = math.pi / 2 + 2 * math.pi * i / max(len(others), 1)
        pos[n] = (1.5 * math.cos(a), 1.2 * math.sin(a))
    return pos


def fig_dag(bn: BNModel, target: str = C.TARGET_NODE, height: int = 460) -> go.Figure:
    pos = _positions(bn.nodes, target)
    fig = go.Figure()
    r = 0.26
    for a, b in sorted(bn.edges):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        d = math.hypot(x1 - x0, y1 - y0) or 1
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        fig.add_annotation(x=x1 - ux * r, y=y1 - uy * r, ax=x0 + ux * r, ay=y0 + uy * r, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2.4,
                           arrowcolor=T["accent"], text="")
        fig.add_trace(go.Scatter(x=[(x0 + x1) / 2], y=[(y0 + y1) / 2], mode="markers", marker=dict(size=14, opacity=0),
                                 hovertemplate=f"<b>{a} → {b}</b><br>parents of {b}: {', '.join(bn.parents_of(b))}<extra></extra>",
                                 showlegend=False))
    xs = [pos[n][0] for n in bn.nodes]
    ys = [pos[n][1] for n in bn.nodes]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=[n.replace("_", "<br>") for n in bn.nodes], textposition="middle center",
        textfont=dict(size=11, color=T["bg"]),
        marker=dict(size=84, color=[T["accent"] if n == target else T["gold"] if bn.parents_of(n) or bn.children_of(n) else "#5A6670"
                                    for n in bn.nodes], line=dict(color=T["text"], width=1.5)),
        customdata=[[", ".join(bn.parents_of(n)) or "none", ", ".join(bn.children_of(n)) or "none"] for n in bn.nodes],
        hovertemplate="<b>%{text}</b><br>parents: %{customdata[0]}<br>children: %{customdata[1]}<extra></extra>", showlegend=False))
    fig.update_xaxes(visible=False, range=[-2.4, 2.4])
    fig.update_yaxes(visible=False, range=[-1.9, 1.9], scaleanchor="x", scaleratio=0.8)
    _layout(fig, "Learned Bayesian-network structure (directed acyclic graph)", height=height, legend=False)
    fig.update_layout(margin=dict(l=10, r=10, t=64, b=10))
    return fig


def fig_bic_progress(history: list[dict], score_name: str = "BIC") -> go.Figure:
    h = pd.DataFrame(history)
    fig = go.Figure(go.Scatter(
        x=h["step"], y=h["total_score"], mode="lines+markers+text", line=dict(color=T["accent"], width=2.5),
        marker=dict(size=10, color=T["accent"]), text=h["operation"].where(h["step"] > 0, "start").str.replace(" (empty graph)", "", regex=False),
        textposition="top center", customdata=np.stack([h["edge"], h["gain"]], axis=1),
        hovertemplate="step %{x}<br>" + score_name + " %{y:.2f}<br>%{customdata[0]}<br>gain %{customdata[1]}<extra></extra>"))
    fig.update_xaxes(dtick=1)
    return _layout(fig, f"Hill-Climb search: network {score_name} score after each accepted move",
                   "Search step", f"Total {score_name} (higher is better)", height=340, legend=False)


def fig_cpd(bn: BNModel, node: str) -> go.Figure:
    df = cpd_frame(bn, node)
    cols = [" | ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    z = df.to_numpy(dtype=float)
    txt = [["n/a" if np.isnan(v) else f"{v:.3f}" for v in r] for r in z]
    fig = go.Figure(go.Heatmap(z=z, x=cols, y=list(df.index), zmin=0, zmax=1, text=txt, texttemplate="%{text}",
                               colorscale=[[0, T["surface"]], [0.5, T["accent_dim"]], [1, T["accent"]]],
                               hovertemplate="P(" + node + "=%{y} | parents=%{x}) = %{z:.3f}<extra></extra>",
                               colorbar=dict(title="Prob.")))
    fig.update_yaxes(autorange="reversed")
    par = ", ".join(bn.parents_of(node)) or "none"
    return _layout(fig, f"CPD of {node} (parents: {par})", "Parent configuration", node, height=320, legend=False)


def fig_conditional_stack(dist_net: pd.Series, dist_emp: Optional[pd.Series], title: str) -> go.Figure:
    cmap = color_map(list(dist_net.index))
    fig = go.Figure()
    rows = {"Learned network (exact)": dist_net}
    if dist_emp is not None and len(dist_emp):
        rows["Matching months (empirical)"] = dist_emp
    for s in dist_net.index:
        fig.add_trace(go.Bar(y=list(rows), x=[float(v.get(s, 0)) for v in rows.values()], name=s, orientation="h",
                             marker_color=cmap[s], text=[f"{float(v.get(s, 0)):.1%}" for v in rows.values()],
                             textposition="inside", hovertemplate=f"{s}: %{{x:.1%}}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    return _layout(fig, title, "Probability", None, height=250)


def fig_vix_conditional(table: pd.DataFrame, names: Sequence[str], cond: str = "VIX_Level") -> go.Figure:
    cmap = color_map(names)
    fig = go.Figure()
    for n in names:
        col = f"P({n})"
        if col not in table:
            continue
        fig.add_trace(go.Bar(x=table[cond], y=table[col], name=n, marker_color=cmap[n],
                             text=[f"{v:.0%}" for v in table[col]], textposition="inside",
                             customdata=table["Months"], hovertemplate=f"P({n} | {cond}=%{{x}}) = %{{y:.1%}}<br>n = %{{customdata}} months<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return _layout(fig, f"Conditional regime probability given {cond}", cond.replace("_", " "), "P(regime | level)", height=360)


# -------------------------------------------------------------- toy lab

def fig_toy_matrix(M: np.ndarray, rows: Sequence[str], cols: Sequence[str], title: str, xtitle: str, ytitle: str,
                   fmt: str = "{:.4f}") -> go.Figure:
    fig = go.Figure(go.Heatmap(z=M, x=list(cols), y=list(rows), colorscale=[[0, T["surface"]], [1, T["accent"]]],
                               text=[[fmt.format(v) for v in r] for r in M], texttemplate="%{text}",
                               hovertemplate="%{y}, %{x}: %{z:.6f}<extra></extra>", colorbar=dict(title="Value")))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, xtitle, ytitle, height=300, legend=False)


def fig_toy_trellis(states: Sequence[str], obs: Sequence[str], path_idx: Sequence[int], delta: np.ndarray) -> go.Figure:
    T_, N = delta.shape
    fig = go.Figure()
    for t in range(T_):
        for j in range(N):
            on = path_idx[t] == j
            fig.add_trace(go.Scatter(x=[t], y=[j], mode="markers+text", text=[f"{delta[t, j]:.4f}"], textposition="bottom center",
                                     marker=dict(size=30, color=T["accent"] if on else T["surface2"],
                                                 line=dict(color=T["accent"] if on else T["border"], width=2)),
                                     showlegend=False, hovertemplate=f"{states[j]} at t={t + 1}<br>δ={delta[t, j]:.6f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(range(T_)), y=list(path_idx), mode="lines", line=dict(color=T["accent"], width=3),
                             showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(tickmode="array", tickvals=list(range(T_)), ticktext=[f"t={t + 1}<br>{o}" for t, o in enumerate(obs)])
    fig.update_yaxes(tickmode="array", tickvals=list(range(N)), ticktext=list(states), autorange="reversed")
    return _layout(fig, "Viterbi trellis — highlighted nodes form the optimal path (value = δ)", "Observation", "Hidden state", height=340, legend=False)


def fig_em_likelihood(ll_hist: Sequence[float]) -> go.Figure:
    fig = go.Figure(go.Scatter(x=list(range(1, len(ll_hist) + 1)), y=list(ll_hist), mode="lines+markers",
                               line=dict(color=T["accent"], width=2.5), hovertemplate="iteration %{x}<br>log P(O|λ) = %{y:.5f}<extra></extra>"))
    return _layout(fig, "Baum–Welch: log-likelihood log P(O | λ) per EM iteration", "EM iteration", "log P(O | λ)", height=320, legend=False)
