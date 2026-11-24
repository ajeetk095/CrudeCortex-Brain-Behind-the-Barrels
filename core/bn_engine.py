"""Bayesian network engine — CrudeCortex GWP2.

Score-based structure learning (Hill Climb Search over add / delete / reverse
operations, BIC score, maximum-parent constraint, DAG validation), maximum
likelihood CPD estimation and *exact inference by enumeration* on the learned
network, for the discretised macro panel (WTI_Regime + four macro nodes).

Correctness notes
-----------------
* ``bic_score`` follows pgmpy's ``bic-d`` definition: log-likelihood minus
  ``0.5 ln N * (r - 1) * q`` where ``r`` is the node cardinality and ``q`` the
  product of the parents' cardinalities.
* The local score of a node always uses the node's **parents** (the sources of
  its incoming edges). An earlier version of this module collected the wrong
  tuple element, so nodes with existing parents were scored on themselves; see
  docs/DATA_AND_DISCREPANCY_REGISTER.md (D4).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import config as C
from .config import BNConfig

REGIME_ORDER = ["Bear", "Stagnant", "Bull"]
LEVEL_ORDER = ["Low", "Medium", "High"]
CATEGORY_ORDERS = {
    "Low": 0, "Medium": 1, "High": 2, "Down": 0, "Flat": 1, "Up": 2,
    "Falling": 0, "Rising": 2, "Bear": 0, "Stagnant": 1, "Bull": 2,
}
TREND_LABELS = {"DXY": ("Down", "Flat", "Up"), "IndPro": ("Falling", "Flat", "Rising"),
                "CPI_Nrg": ("Falling", "Flat", "Rising")}


_TIE = 1e-9   # scores closer than this are ties (earlier candidate in search order wins)


class InferenceError(ValueError):
    """Invalid or impossible evidence / query."""


# --------------------------------------------------------------------------
# Discretisation and panel construction
# --------------------------------------------------------------------------

def discretization_thresholds(s: pd.Series, method: str = "report") -> tuple[float, float]:
    """Lower/upper cut-points used by :func:`discretize`."""
    if method == "tertile":
        return float(s.quantile(1 / 3)), float(s.quantile(2 / 3))
    return float(s.quantile(0.33)), float(s.quantile(0.67))


def discretize(s: pd.Series, labels=("Low", "Medium", "High"), method: str = "report") -> pd.Series:
    """Quantile discretisation into three classes (plain ``str``).

    ``method="report"`` reproduces the notebook: cut at the 33rd and 67th
    percentiles (``v < q33`` -> low, ``v < q67`` -> medium, else high).
    ``method="tertile"`` uses pandas ``qcut`` equal-frequency bins (the earlier
    app variant). Missing values stay missing.
    """
    if method == "tertile":
        return pd.qcut(s, q=3, labels=list(labels)).astype(object).where(s.notna()).map(
            lambda v: v if pd.isna(v) else str(v))
    q33, q67 = discretization_thresholds(s, "report")
    out = pd.Series(np.where(s < q33, labels[0], np.where(s < q67, labels[1], labels[2])),
                    index=s.index, dtype=object)
    return out.where(s.notna())


def build_discrete_panel(feature_df: pd.DataFrame, labels: pd.Series, df: pd.DataFrame,
                         method: str = "report", variables: str = "trend",
                         indicators: Iterable[str] = C.MACRO_KEYS) -> pd.DataFrame:
    """Discrete dataset: regime + discretised macro nodes, indexed by month.

    ``variables="level"``: every macro series is discretised on its *level*
    (nodes ``VIX_Level``, ``IndPro_Level``, ``CPI_Level``, ``DXY_Level``).
    ``variables="trend"``: the executed notebook's definition — VIX level plus
    month-on-month changes for the others (``DXY_Trend`` ...).
    """
    indicators = [k for k in C.MACRO_KEYS if k in set(indicators)]
    names = C.BN_LEVEL_NODES if variables == "level" else C.BN_TREND_NODES
    idx = feature_df.index
    panel = pd.DataFrame({C.TARGET_NODE: labels.reindex(idx)})
    for key in indicators:
        if variables == "trend" and key != "VIX":
            if key == "CPI_Nrg":
                x = df["CPI_Nrg"].pct_change()
            else:
                x = np.log(df[key] / df[key].shift(1))
            panel[names[key]] = discretize(x.reindex(idx), TREND_LABELS[key], method)
        else:
            panel[names[key]] = discretize(df.loc[idx, key], LEVEL_ORDER, method)
    return panel.dropna()


def category_order(node: str, states: Iterable[str]) -> list[str]:
    return sorted(states, key=lambda s: (CATEGORY_ORDERS.get(s, 99), str(s)))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _nz(series: pd.Series) -> pd.Series:
    vc = series.value_counts()
    return vc[vc > 0]


def _penalty(score: str, n: int, k: int) -> float:
    return (0.5 * math.log(n) * k) if score == "bic" else float(k)


def bic_score(data: pd.DataFrame, node: str, parents: tuple = (), score: str = "bic") -> float:
    """Local decomposable score of ``node`` given ``parents`` (higher is better).

    ``score="bic"``: LL - (ln N / 2) k ; ``score="aic"``: LL - k, where
    ``k = (r - 1) * q`` free parameters.
    """
    n = len(data)
    parents = tuple(parents)
    if not parents:
        counts = _nz(data[node])
        ll = float(sum(c * math.log(c / n) for c in counts))
        return ll - _penalty(score, n, len(counts) - 1)
    ll = 0.0
    for _, grp in data.groupby(list(parents), observed=True)[node]:
        nj = len(grp)
        for c in _nz(grp):
            ll += c * math.log(c / nj)
    q = 1
    for p in parents:
        q *= data[p].nunique()
    return ll - _penalty(score, n, (data[node].nunique() - 1) * q)


class ScoreCache:
    """Memoised local scores (many candidate moves share the same (node, parents))."""

    def __init__(self, data: pd.DataFrame, score: str = "bic"):
        self.data, self.score = data, score
        self._memo: dict[tuple[str, frozenset], float] = {}
        self.evaluations = 0

    def __call__(self, node: str, parents: Iterable[str]) -> float:
        key = (node, frozenset(parents))
        if key not in self._memo:
            self._memo[key] = bic_score(self.data, node, tuple(sorted(key[1])), self.score)
            self.evaluations += 1
        return self._memo[key]


def total_score(data: pd.DataFrame, edges: set, nodes: list, score: str = "bic") -> float:
    """Decomposable network score of a DAG."""
    sc = ScoreCache(data, score)
    return float(sum(sc(n, [a for a, b in edges if b == n]) for n in nodes))


# --------------------------------------------------------------------------
# Graph utilities
# --------------------------------------------------------------------------

def is_dag(edges: set, nodes: list) -> bool:
    """Kahn's algorithm acyclicity check."""
    adj = {n: set() for n in nodes}
    indeg = {n: 0 for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        indeg[b] += 1
    queue = [n for n in nodes if indeg[n] == 0]
    seen = 0
    while queue:
        n = queue.pop()
        seen += 1
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    return seen == len(nodes)


def topological_order(edges: set, nodes: list) -> list:
    adj = {n: set() for n in nodes}
    indeg = {n: 0 for n in nodes}
    for a, b in edges:
        adj[a].add(b)
        indeg[b] += 1
    queue = sorted(n for n in nodes if indeg[n] == 0)
    out = []
    while queue:
        n = queue.pop(0)
        out.append(n)
        for m in sorted(adj[n]):
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if len(out) != len(nodes):
        raise ValueError("Graph contains a cycle.")
    return out


# --------------------------------------------------------------------------
# Hill Climb Search
# --------------------------------------------------------------------------

def hill_climb_search(data: pd.DataFrame, nodes: list, max_parents: int = 2,
                      score: str = "bic", min_gain: float = 1e-9, max_iter: int = 500):
    """Greedy search over add/delete/reverse operations maximising the network score.

    Returns ``(edges, history)``. ``history`` has one row per accepted move
    (step, operation, edge, gain, total score, n_edges) starting with the empty graph.
    """
    nodes = list(nodes)
    sc = ScoreCache(data, score)
    edges: set = set()

    def parents(n: str, e: set) -> list:
        return [a for a, b in e if b == n]

    def total(e: set) -> float:
        return sum(sc(n, parents(n, e)) for n in nodes)

    history = [{"step": 0, "operation": "init (empty graph)", "edge": "—", "gain": 0.0,
                "total_score": round(total(edges), 3), "n_edges": 0}]
    for step in range(1, max_iter + 1):
        best: Optional[tuple[float, str, tuple]] = None
        for x in nodes:
            for y in nodes:
                if x == y:
                    continue
                pa_y = parents(y, edges)
                if (x, y) not in edges:
                    if len(pa_y) < max_parents and is_dag(edges | {(x, y)}, nodes):
                        gain = sc(y, pa_y + [x]) - sc(y, pa_y)
                        if best is None or gain > best[0] + _TIE:
                            best = (gain, "add", (x, y))
                else:
                    # delete x -> y
                    gain = sc(y, [p for p in pa_y if p != x]) - sc(y, pa_y)
                    if best is None or gain > best[0] + _TIE:
                        best = (gain, "delete", (x, y))
                    # reverse x -> y  into  y -> x
                    rev = (edges - {(x, y)}) | {(y, x)}
                    pa_x = parents(x, edges)
                    if len(pa_x) < max_parents and is_dag(rev, nodes):
                        g2 = (sc(y, [p for p in pa_y if p != x]) - sc(y, pa_y)
                              + sc(x, pa_x + [y]) - sc(x, pa_x))
                        if best is None or g2 > best[0] + _TIE:
                            best = (g2, "reverse", (x, y))
        if best is None or best[0] <= min_gain:
            break
        gain, op, (x, y) = best
        if op == "add":
            edges.add((x, y))
        elif op == "delete":
            edges.discard((x, y))
        else:
            edges.discard((x, y))
            edges.add((y, x))
        history.append({"step": step, "operation": op, "edge": f"{x} → {y}" if op != "reverse" else f"{x} → {y} (reversed to {y} → {x})",
                        "gain": round(float(gain), 4), "total_score": round(total(edges), 3),
                        "n_edges": len(edges)})
    assert is_dag(edges, nodes)
    return edges, history


# --------------------------------------------------------------------------
# Model, CPDs, inference
# --------------------------------------------------------------------------

@dataclass
class BNModel:
    data: pd.DataFrame
    nodes: list
    edges: set
    history: list
    cpds: dict = field(default_factory=dict)
    max_parents: int = 2
    score: str = "bic"
    states: dict = field(default_factory=dict)
    unseen: dict = field(default_factory=dict)       # node -> list of unobserved parent configs

    def parents_of(self, node: str) -> list:
        return sorted([a for a, b in self.edges if b == node])

    def children_of(self, node: str) -> list:
        return sorted([b for a, b in self.edges if a == node])


def fit_cpds(bn: BNModel) -> BNModel:
    """Maximum-likelihood conditional probability tables (relative frequencies)."""
    for node in bn.nodes:
        bn.states[node] = category_order(node, bn.data[node].unique())
    for node in bn.nodes:
        parents = bn.parents_of(node)
        if not parents:
            counts = _nz(bn.data[node])
            probs = counts / counts.sum()
            bn.cpds[node] = {"parents": [], "table": {(): probs.to_dict()},
                             "counts": {(): int(counts.sum())}}
            bn.unseen[node] = []
        else:
            table, support = {}, {}
            for keys, grp in bn.data.groupby(parents, observed=True):
                key = keys if isinstance(keys, tuple) else (keys,)
                pr = _nz(grp[node])
                table[key] = (pr / pr.sum()).to_dict()
                support[key] = int(len(grp))
            bn.cpds[node] = {"parents": parents, "table": table, "counts": support}
            all_cfgs = list(itertools.product(*[bn.states[p] for p in parents]))
            bn.unseen[node] = [c for c in all_cfgs if c not in table]
    return bn


def _cpd_prob(bn: BNModel, node: str, state: str, parent_cfg: tuple) -> float:
    cpd = bn.cpds[node]
    row = cpd["table"].get(parent_cfg)
    if row is None:                    # unobserved parent configuration: uniform fallback
        return 1.0 / len(bn.states[node])
    return float(row.get(state, 0.0))


def validate_evidence(bn: BNModel, target: str, evidence: Optional[dict]) -> dict:
    """Return a cleaned evidence dict or raise :class:`InferenceError`."""
    if target not in bn.nodes:
        raise InferenceError(f"Unknown target variable '{target}'. Valid: {', '.join(bn.nodes)}.")
    if evidence is None:
        return {}
    if not isinstance(evidence, dict):
        raise InferenceError("Evidence must be a mapping {variable: category}.")
    clean = {}
    for k, v in evidence.items():
        if k not in bn.nodes:
            raise InferenceError(f"Unsupported evidence variable '{k}'. Valid: {', '.join(bn.nodes)}.")
        if k == target:
            raise InferenceError("The target variable cannot also be evidence.")
        if v not in bn.states[k]:
            raise InferenceError(f"'{v}' is not a category of {k}. Valid: {', '.join(bn.states[k])}.")
        clean[k] = v
    return clean


def infer_network(bn: BNModel, target: str, evidence: Optional[dict] = None) -> pd.Series:
    """Exact P(target | evidence) by enumeration of the network's joint distribution.

    The joint is the product of the CPDs implied by the learned DAG, so evidence
    on a variable that is *not* connected to the target (in the learned graph)
    leaves the answer at the target's marginal/CPD. Raises
    :class:`InferenceError` if the evidence has zero probability.
    """
    ev = validate_evidence(bn, target, evidence)
    free = [n for n in bn.nodes if n != target and n not in ev]
    post = {s: 0.0 for s in bn.states[target]}
    for t_state in bn.states[target]:
        total = 0.0
        for combo in itertools.product(*[bn.states[n] for n in free]):
            assign = {**ev, **dict(zip(free, combo)), target: t_state}
            p = 1.0
            for n in bn.nodes:
                pa = tuple(assign[q] for q in bn.cpds[n]["parents"])
                p *= _cpd_prob(bn, n, assign[n], pa)
                if p == 0.0:
                    break
            total += p
        post[t_state] = total
    z = sum(post.values())
    if z <= 0:
        raise InferenceError("The evidence has zero probability under the learned network.")
    return pd.Series({k: v / z for k, v in post.items()}, name=f"P({target}|evidence)")


def infer_empirical(bn: BNModel, target: str, evidence: Optional[dict] = None) -> tuple[pd.Series, int]:
    """Relative frequency of ``target`` among rows matching the evidence (data-only, no network).

    Returns ``(distribution, n_matching_rows)``; an empty Series if no row matches.
    """
    ev = validate_evidence(bn, target, evidence)
    mask = pd.Series(True, index=bn.data.index)
    for k, v in ev.items():
        mask &= bn.data[k] == v
    sub = bn.data.loc[mask, target]
    if len(sub) == 0:
        return pd.Series(dtype=float), 0
    vc = sub.value_counts(normalize=True).reindex(bn.states[target]).fillna(0.0)
    return vc, int(len(sub))


def infer_node(bn: BNModel, node: str, evidence: dict) -> pd.Series:
    """Backward-compatible empirical conditional (rows matching the evidence)."""
    return infer_empirical(bn, node, evidence)[0]


def build_bn(panel: pd.DataFrame, max_parents: int = 2, score: str = "bic",
             target: str = C.TARGET_NODE) -> BNModel:
    """Learn structure (Hill Climb) and fit CPDs.

    Orientation convention: two orientations of the same link are *score
    equivalent* (same BIC, same joint distribution), so the data cannot choose
    between them. The search lists ``target`` last so that ties are broken by
    orienting links *into* the target (e.g. ``VIX_Level -> WTI_Regime``, the
    reading used in the report). This is a convention, not a causal finding.
    """
    nodes = list(panel.columns)
    search_order = [n for n in nodes if n != target] + ([target] if target in nodes else [])
    edges, history = hill_climb_search(panel, search_order, max_parents=max_parents, score=score)
    bn = BNModel(panel, nodes, edges, history, max_parents=max_parents, score=score)
    return fit_cpds(bn)


# --------------------------------------------------------------------------
# Tables for display / export
# --------------------------------------------------------------------------

def edge_table(bn: BNModel) -> pd.DataFrame:
    """Learned edges with the score loss incurred if each edge were removed."""
    sc = ScoreCache(bn.data, bn.score)
    rows = []
    for a, b in sorted(bn.edges):
        pa = bn.parents_of(b)
        rows.append({"Parent": a, "Child": b,
                     "Score loss if removed": round(sc(b, pa) - sc(b, [p for p in pa if p != a]), 3),
                     "Child parents": ", ".join(pa)})
    return pd.DataFrame(rows, columns=["Parent", "Child", "Score loss if removed", "Child parents"])


def cpd_frame(bn: BNModel, node: str) -> pd.DataFrame:
    """CPD as a matrix: rows = node states, columns = parent configurations."""
    cpd = bn.cpds[node]
    states = bn.states[node]
    parents = cpd["parents"]
    if not parents:
        col = pd.Index(["(no parents)"])
        vals = np.array([[cpd["table"][()].get(s, 0.0)] for s in states])
    else:
        cfgs = list(itertools.product(*[bn.states[p] for p in parents]))
        col = pd.MultiIndex.from_tuples(cfgs, names=parents) if len(parents) > 1 else pd.Index(
            [c[0] for c in cfgs], name=parents[0])
        vals = np.array([[(cpd["table"][c].get(s, 0.0) if c in cpd["table"] else np.nan)
                          for c in cfgs] for s in states])
    return pd.DataFrame(vals, index=pd.Index(states, name=node), columns=col)


def cpds_long(bn: BNModel) -> pd.DataFrame:
    """Tidy CPD table for export."""
    rows = []
    for node in bn.nodes:
        cpd = bn.cpds[node]
        for cfg, dist in cpd["table"].items():
            for s in bn.states[node]:
                rows.append({"node": node, "parents": ", ".join(cpd["parents"]) or "(none)",
                             "parent_config": ", ".join(cfg) if cfg else "(none)",
                             "state": s, "probability": float(dist.get(s, 0.0)),
                             "rows_supporting_config": cpd["counts"][cfg]})
    return pd.DataFrame(rows)


def conditional_regime_table(panel: pd.DataFrame, target: str = "WTI_Regime",
                             cond: str = "VIX_Level") -> pd.DataFrame:
    """Report Table 14 — empirical P(regime | VIX level) from MLE frequencies."""
    states = category_order(target, panel[target].unique())
    rows = []
    for lvl in category_order(cond, panel[cond].unique()):
        sub = panel[panel[cond] == lvl][target]
        n = len(sub)
        if n == 0:
            continue
        vc = sub.value_counts(normalize=True)
        row = {cond: lvl, "Months": int(n)}
        for s in states:
            row[f"P({s})"] = round(float(vc.get(s, 0)), 3)
        row["Dominant regime"] = vc.idxmax()
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Chronological hold-out (report Step 9B: 80 % train / 20 % test)
# --------------------------------------------------------------------------

def holdout_evaluation(panel: pd.DataFrame, edges: set, target: str = C.TARGET_NODE,
                       train_frac: float = 0.8, alpha: float = 1.0) -> dict:
    """Fit CPDs on the first 80 % of months and score the last 20 %.

    Predictive distribution = P(target | target's parents) from the training
    CPD with Laplace smoothing ``alpha`` (so unseen combinations never give
    zero probability). Compared with the training-sample marginal baseline.
    This is a *descriptive* in-sample-structure check — the structure itself was
    learned on the full sample, as in the report.
    """
    n = len(panel)
    split = int(n * train_frac)
    train, test = panel.iloc[:split], panel.iloc[split:]
    if len(test) == 0 or len(train) == 0:
        raise ValueError("Hold-out split needs at least a few observations.")
    states = category_order(target, panel[target].unique())
    parents = sorted(a for a, b in edges if b == target)

    def dist(df_: pd.DataFrame, row: pd.Series | None) -> np.ndarray:
        sub = df_ if row is None else df_[np.logical_and.reduce([df_[p] == row[p] for p in parents])]
        cnt = np.array([(sub[target] == s).sum() for s in states], dtype=float) + alpha
        return cnt / cnt.sum()

    base = dist(train, None)
    ll_m = ll_b = 0.0
    hit_m = hit_b = 0
    for _, row in test.iterrows():
        pm = dist(train, row) if parents else base
        k = states.index(row[target])
        ll_m += -math.log(pm[k])
        ll_b += -math.log(base[k])
        hit_m += int(np.argmax(pm) == k)
        hit_b += int(np.argmax(base) == k)
    nt = len(test)
    return {"n_train": len(train), "n_test": nt,
            "train_period": f"{train.index[0]:%Y-%m} – {train.index[-1]:%Y-%m}" if hasattr(train.index[0], "year") else "",
            "test_period": f"{test.index[0]:%Y-%m} – {test.index[-1]:%Y-%m}" if hasattr(test.index[0], "year") else "",
            "target_parents": parents, "alpha": alpha,
            "log_loss_model": ll_m / nt, "log_loss_baseline": ll_b / nt,
            "accuracy_model": hit_m / nt, "accuracy_baseline": hit_b / nt,
            "train_marginal": dict(zip(states, base.tolist()))}


def train_conditional(panel: pd.DataFrame, target: str, evidence: dict, train_frac: float = 0.8) -> pd.Series:
    """Unsmoothed relative frequency of ``target`` given evidence within the training slice."""
    train = panel.iloc[: int(len(panel) * train_frac)]
    mask = pd.Series(True, index=train.index)
    for k, v in evidence.items():
        mask &= train[k] == v
    sub = train.loc[mask, target]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.value_counts(normalize=True).reindex(category_order(target, panel[target].unique())).fillna(0.0)
