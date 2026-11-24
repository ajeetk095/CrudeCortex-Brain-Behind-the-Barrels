"""GWP2 data pipeline — CrudeCortex.

Builds the monthly macro panel and the stationary 2-column feature matrix
(WTI monthly log-return, 12-month annualised rolling volatility, standardised)
used by the Gaussian HMM, exactly as in the submitted report/notebook, while
making every date and count *derive from the data actually loaded*.

Why 228 x 5 and 216 x 2 (reconciled against report and notebook)
-----------------------------------------------------------------
* The source window is 2005-01 -> 2024-12 (240 months for WTI), but
  ``DTWEXBGS`` (broad dollar index) only begins on 2006-01-02, so the inner-join
  master panel starts 2006-01 -> 228 monthly rows.
* A monthly log-return needs the previous month (1 row) and a 12-return rolling
  standard deviation needs 11 further rows, so 12 warm-up rows are consumed:
  features run 2007-01 -> 2024-12 = 216 rows.
* The report text that says "216 months from January 2006" mislabels the
  start; the modelled sample starts January 2007. See docs/DATA_AND_DISCREPANCY_REGISTER.md.

Public helpers keep backward compatibility with the original module:
``to_monthly``, ``build_master``, ``build_features``, ``get_gwp2_data``.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as C
from .config import DataConfig
from .data_sources import (DataError, DataValidationError, SeriesLoad, load_all)

__all__ = ["Dataset", "build_dataset", "get_dataset", "get_gwp2_data", "to_monthly",
           "build_master", "build_features", "DateRangeError", "available_bounds"]


class DateRangeError(DataError):
    """Requested date window is outside what the data (plus warm-up) can support."""


# --------------------------------------------------------------------------
# Resampling helpers
# --------------------------------------------------------------------------

def to_monthly(series: pd.Series, how: str = "last") -> pd.Series:
    """Resample to month-start frequency; interior gaps forward-filled (as in the notebook)."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    r = s.resample("MS")
    out = r.mean() if how == "mean" else r.last()
    return out.ffill()


def _drop_incomplete_last_month(daily: pd.Series) -> pd.Series:
    """Remove the trailing month if the daily series stops well before month-end.

    A partial month would otherwise be reported as a (wrong) 'month-end close'.
    """
    if daily.empty:
        return daily
    last = daily.index[-1]
    month_end = last + pd.offsets.MonthEnd(0)
    if (month_end - last).days > 4:
        return daily[daily.index < last.to_period("M").to_timestamp()]
    return daily


def _vix_monthly(load: SeriesLoad, aggregation: str) -> tuple[pd.Series, str, Optional[str]]:
    """Monthly VIX per the requested aggregation. Returns (series, effective_aggregation, note)."""
    s = load.values
    if load.frequency == "monthly":
        note = None
        if aggregation == "monthly_mean":
            note = ("Monthly-mean VIX was requested but only monthly closes are available "
                    "(Yahoo fallback); month-end close is used instead.")
        return s.resample("MS").last(), "month_end_close", note
    daily = _drop_incomplete_last_month(s)
    if aggregation == "monthly_mean":
        return daily.resample("MS").mean(), "monthly_mean", None
    return daily.resample("MS").last(), "month_end_close", None


# --------------------------------------------------------------------------
# Master panel and features
# --------------------------------------------------------------------------

def _assemble_full_master(loads: dict[str, SeriesLoad], vix_aggregation: str
                          ) -> tuple[pd.DataFrame, dict, list[str], str]:
    """Inner-join all five monthly series over their common span (no date clipping)."""
    warnings_: list[str] = []
    cols: dict[str, pd.Series] = {}
    filled: dict[str, pd.Series] = {}
    for key, spec in C.SERIES.items():
        ld = loads[spec.series_id]
        if key == "VIX":
            m, eff_agg, note = _vix_monthly(ld, vix_aggregation)
            if note:
                warnings_.append(note)
        else:
            raw_m = ld.values
            if ld.frequency == "daily":
                raw_m = _drop_incomplete_last_month(raw_m)
            r = raw_m.resample("MS").last()
            filled[key] = r.isna()                 # months with no observation (to be forward-filled)
            m = r.ffill()
        cols[key] = m.rename(key)
    filled["VIX"] = cols["VIX"].isna()
    cols["VIX"] = cols["VIX"].ffill()
    panel = pd.DataFrame(cols)
    joined = panel.dropna()
    if joined.empty:
        raise DataValidationError("The five series share no overlapping months; check the data sources.")
    joined.index.name = "date"
    return joined, filled, warnings_, eff_agg


def available_bounds(full_master: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Earliest/latest *feature* month the data can support (after 12 warm-up rows)."""
    return full_master.index[C.WARMUP_MONTHS], full_master.index[-1]


def build_features(df: pd.DataFrame):
    """Stationary features: WTI log-return and 12M annualised volatility.

    Returns ``(feature_df, X_std, scaler, adf)`` — identical contract to the
    original module.
    """
    if (df["WTI"] <= 0).any():
        raise DataValidationError("WTI contains non-positive prices; log returns are undefined.")
    out = df.copy()
    out["WTI_ret"] = np.log(out["WTI"] / out["WTI"].shift(1))
    out["WTI_vol12"] = out["WTI_ret"].rolling(C.ROLLING_WINDOW).std() * np.sqrt(12)
    feature_df = out[["WTI_ret", "WTI_vol12"]].dropna()
    if len(feature_df) < 30:
        raise DateRangeError(
            f"Only {len(feature_df)} valid feature months in the selected window; "
            "at least 30 are needed to fit a Gaussian HMM.")
    from statsmodels.tsa.stattools import adfuller
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, p_val, lags, nobs, crit, _ = adfuller(feature_df["WTI_ret"], autolag="AIC")
    adf = {"stat": float(adf_stat), "p": float(p_val), "crit_1pct": float(crit["1%"]),
           "crit_5pct": float(crit["5%"]), "lags": int(lags), "nobs": int(nobs)}
    scaler = StandardScaler()
    X_std = scaler.fit_transform(feature_df.values)
    return feature_df, X_std, scaler, adf


def build_master(loads: Optional[dict[str, SeriesLoad]] = None,
                 cfg: Optional[DataConfig] = None) -> pd.DataFrame:
    """Master panel (WTI, IndPro, CPI_Nrg, DXY, VIX) for the configured window."""
    cfg = cfg or DataConfig()
    loads = loads or load_all(cfg.data_mode)
    return build_dataset(loads, cfg).master


@dataclass
class Dataset:
    """Everything downstream modules need, with provenance and window explanation."""
    config: DataConfig
    master: pd.DataFrame            # warm-up + modelled window (default 228 x 5)
    full_master: pd.DataFrame       # all aligned months available (for bounds/controls)
    features: pd.DataFrame          # WTI_ret, WTI_vol12 (default 216 x 2)
    X_std: np.ndarray
    scaler: StandardScaler
    adf: dict
    loads: dict[str, SeriesLoad]
    status: pd.DataFrame
    window: dict
    warnings: list[str] = field(default_factory=list)
    vix_aggregation_effective: str = "month_end_close"
    fingerprint: str = ""

    # convenience
    @property
    def first_feature_month(self) -> pd.Timestamp:
        return self.features.index[0]

    @property
    def last_feature_month(self) -> pd.Timestamp:
        return self.features.index[-1]

    @property
    def is_report_window(self) -> bool:
        return (self.config.feature_start == C.REPORT_FIRST_FEATURE_MONTH
                and self.config.feature_end == C.REPORT_LAST_MONTH)

    def fmt_range(self, which: str = "features") -> str:
        idx = self.features.index if which == "features" else self.master.index
        return f"{idx[0]:%b %Y} – {idx[-1]:%b %Y}"


def _origin_label(ld: SeriesLoad) -> tuple[str, str]:
    if ld.origin == "live":
        return "Live FRED", f"retrieved {ld.retrieved_at}"
    if ld.origin == "cache":
        return "Runtime cache (last successful download)", f"retrieved {ld.retrieved_at or 'unknown'}"
    if ld.origin == "yahoo_fallback":
        return "Yahoo Finance fallback", f"retrieved {ld.retrieved_at}"
    return "Bundled snapshot", f"file dated {ld.file_modified_at or 'unknown'} (not a live retrieval)"


def _status_table(loads: dict[str, SeriesLoad], master: pd.DataFrame, filled: dict[str, pd.Series],
                  vix_eff: str) -> pd.DataFrame:
    rows = []
    for key, spec in C.SERIES.items():
        ld = loads[spec.series_id]
        s = ld.values
        origin, when = _origin_label(ld)
        status = "OK" if ld.origin in ("snapshot", "live") else "OK (fallback in use)"
        if ld.live_attempted and ld.origin != "live" and ld.live_error:
            status = "Live retrieval failed — using " + origin.lower()
        freq = spec.native_frequency
        if key == "VIX" and ld.frequency == "daily":
            freq = f"Daily → monthly ({'month-end close' if vix_eff == 'month_end_close' else 'monthly mean'})"
        elif key == "DXY":
            freq = "Daily → monthly (month-end value)"
        elif ld.frequency == "monthly" and key == "VIX":
            freq = "Monthly (Yahoo close)"
        rows.append({
            "Indicator": spec.label, "Key": key, "Description": spec.description,
            "Provider": spec.provider if ld.origin != "yahoo_fallback" else "Yahoo Finance (^VIX)",
            "Series ID": spec.series_id if ld.origin != "yahoo_fallback" else "^VIX",
            "Frequency": freq, "Units": spec.units,
            "Start date": f"{s.index[0]:%Y-%m-%d}", "End date": f"{s.index[-1]:%Y-%m-%d}",
            "Observations": int(len(s)),
            "Missing values": int(ld.n_placeholders),
            "Months forward-filled": int(filled[key].reindex(master.index).fillna(False).sum()),
            "Months in panel": int(master[key].notna().sum()),
            "Latest observation": f"{s.index[-1]:%Y-%m-%d}: {s.iloc[-1]:,.2f}",
            "Last retrieval": when, "Origin": origin, "Status": status,
            "Notes": "; ".join(ld.notes + ([ld.live_error] if ld.live_error and ld.origin != "live" else [])) or "—",
        })
    return pd.DataFrame(rows)


def _window_explanation(loads: dict[str, SeriesLoad], full_master: pd.DataFrame, master: pd.DataFrame,
                        features: pd.DataFrame, cfg: DataConfig) -> dict:
    """Derive the date-window narrative from the data (no hard-coded dates)."""
    first_valid = {}
    last_valid = {}
    for key, spec in C.SERIES.items():
        s = loads[spec.series_id].values
        first_valid[key] = s.index[0]
        last_valid[key] = s.index[-1]
    src_start, src_end = pd.Timestamp(C.START), pd.Timestamp(C.END)
    in_src = {k: loads[C.SERIES[k].series_id].values for k in C.SERIES}
    first_in_source_window = {
        k: v[(v.index >= src_start) & (v.index <= src_end)].index.min() for k, v in in_src.items()}
    binding_start = max(first_in_source_window, key=lambda k: first_in_source_window[k])
    wti_months = int(((pd.Series(in_src["WTI"]).index >= src_start) &
                      (pd.Series(in_src["WTI"]).index <= src_end)).sum())
    return {
        "source_window": f"{C.START[:7]} → {C.END[:7]}",
        "wti_months_in_source_window": wti_months,
        "first_valid_in_source_window": {k: f"{v:%Y-%m-%d}" for k, v in first_in_source_window.items()},
        "binding_start_series": binding_start,
        "binding_start_series_id": C.SERIES[binding_start].series_id,
        "panel_first": f"{master.index[0]:%Y-%m}", "panel_last": f"{master.index[-1]:%Y-%m}",
        "panel_rows": len(master),
        "feature_first": f"{features.index[0]:%Y-%m}", "feature_last": f"{features.index[-1]:%Y-%m}",
        "feature_rows": len(features),
        "warmup_rows": len(master) - len(features),
        "warmup_explained": ("1 month lost to the first log-return (needs the prior month) + "
                             f"{C.ROLLING_WINDOW - 1} further months until the {C.ROLLING_WINDOW}-return "
                             "rolling window is full."),
        "available_feature_range": f"{available_bounds(full_master)[0]:%Y-%m} → {available_bounds(full_master)[1]:%Y-%m}",
    }


def build_dataset(loads: dict[str, SeriesLoad], cfg: Optional[DataConfig] = None) -> Dataset:
    """Assemble the monthly panel, features, provenance table and window explanation."""
    cfg = cfg or DataConfig()
    full, filled, warns, eff_agg = _assemble_full_master(loads, cfg.vix_aggregation)
    first_ok, last_ok = available_bounds(full)
    fs, fe = pd.Timestamp(cfg.feature_start), pd.Timestamp(cfg.feature_end)
    fs, fe = fs.to_period("M").to_timestamp(), fe.to_period("M").to_timestamp()
    if fs < first_ok:
        raise DateRangeError(
            f"Start {fs:%Y-%m} is earlier than the first modellable month {first_ok:%Y-%m}: "
            f"the master panel begins {full.index[0]:%Y-%m} (limited by "
            f"{_binding(loads)}) and {C.WARMUP_MONTHS} warm-up months are consumed by returns "
            "and the rolling volatility window.")
    if fe > last_ok:
        raise DateRangeError(f"End {fe:%Y-%m} is after the last complete month in the data ({last_ok:%Y-%m}).")
    if fe <= fs:
        raise DateRangeError("The end month must be after the start month.")

    master_start = fs - pd.DateOffset(months=C.WARMUP_MONTHS)
    master = full.loc[(full.index >= master_start) & (full.index <= fe)].copy()
    expected = pd.date_range(master.index[0], master.index[-1], freq="MS")
    if len(expected) != len(master):
        missing = expected.difference(master.index)
        raise DataValidationError(
            "The monthly panel has gaps inside the selected window: "
            + ", ".join(f"{d:%Y-%m}" for d in missing[:6]) + (" …" if len(missing) > 6 else ""))
    features, X_std, scaler, adf = build_features(master)   # first feature row == fs by construction
    status = _status_table(loads, master, filled, eff_agg)
    window = _window_explanation(loads, full, master, features, cfg)
    fp = hashlib.sha256(pd.util.hash_pandas_object(master, index=True).values.tobytes()).hexdigest()[:16]
    for ld in loads.values():
        if ld.live_attempted and ld.origin != "live" and ld.live_error and cfg.data_mode == "live":
            warns.append(f"{ld.series_id}: live retrieval failed ({ld.live_error}); "
                         f"using {ld.origin}.")
    return Dataset(cfg, master, full, features, X_std, scaler, adf, loads, status, window,
                   warns, eff_agg, fp)


def _binding(loads: dict[str, SeriesLoad]) -> str:
    firsts = {k: loads[s.series_id].values.index[0] for k, s in C.SERIES.items()}
    k = max(firsts, key=lambda x: firsts[x])
    return f"{C.SERIES[k].series_id} (first observation {firsts[k]:%Y-%m-%d})"


def get_dataset(cfg: Optional[DataConfig] = None, **load_kw) -> Dataset:
    """Load providers according to ``cfg.data_mode`` and build the dataset."""
    cfg = cfg or DataConfig()
    return build_dataset(load_all(cfg.data_mode, **load_kw), cfg)


def get_gwp2_data():
    """Backward-compatible tuple ``(df, feature_df, X_std, scaler, adf)`` (report defaults)."""
    ds = get_dataset(DataConfig())
    return ds.master, ds.features, ds.X_std, ds.scaler, ds.adf


def vix_method_comparison(ds: Dataset) -> Optional[pd.DataFrame]:
    """Month-end close vs monthly mean of daily VIX over the modelled window.

    Returns ``None`` when only monthly bars are available (Yahoo fallback).
    """
    ld = ds.loads["VIXCLS"]
    if ld.frequency != "daily":
        return None
    daily = _drop_incomplete_last_month(ld.values)
    last, mean = daily.resample("MS").last(), daily.resample("MS").mean()
    out = pd.DataFrame({"Month-end close": last, "Monthly mean": mean}).reindex(ds.features.index)
    out["Difference (close - mean)"] = out["Month-end close"] - out["Monthly mean"]
    return out
