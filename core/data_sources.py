"""Data acquisition for CrudeCortex GWP2 (framework-free, unit-testable).

Provider order for every FRED series
------------------------------------
* ``mode="snapshot"`` (default, reproducible): bundled ``data/fred/<ID>.csv``
  -> runtime cache -> live FRED (only if nothing local exists).
* ``mode="live"``: live FRED (timeout + retries) -> runtime cache (last
  successful download) -> bundled snapshot. The origin actually used is always
  reported in :class:`SeriesLoad` so the UI never hides a substitution.

Yahoo Finance (^VIX monthly close) is used **only** as the last resort for VIX
when no FRED copy of VIXCLS can be obtained from any route.

No API key is needed: the public ``fredgraph.csv`` endpoint is used.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

from . import config as C


class DataError(Exception):
    """Base class for data-layer errors (always carries a user-readable message)."""


class DataUnavailableError(DataError):
    """No provider, cache or snapshot could supply the series."""


class DataValidationError(DataError):
    """The payload was retrieved but is not a valid numeric time series."""


Origin = Literal["live", "cache", "snapshot", "yahoo_fallback"]


@dataclass
class SeriesLoad:
    """A validated series together with its provenance."""
    series_id: str
    values: pd.Series
    origin: Origin
    frequency: str                       # "daily" | "monthly" (native, as delivered)
    retrieved_at: Optional[str] = None   # ISO-8601 UTC when downloaded; None for snapshot
    file_modified_at: Optional[str] = None
    n_raw_rows: int = 0
    n_placeholders: int = 0              # '.' / non-numeric rows dropped
    n_bad_dates: int = 0
    n_duplicates: int = 0
    live_attempted: bool = False
    live_error: Optional[str] = None
    path: Optional[str] = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Parsing & validation
# --------------------------------------------------------------------------

def parse_fred_csv_text(text: str, series_id: str) -> tuple[pd.Series, dict]:
    """Parse and validate the text of a FRED ``fredgraph.csv`` payload.

    Returns ``(series, stats)`` where *series* has a sorted, unique,
    timezone-naive DatetimeIndex and float values, and *stats* counts the rows
    that were removed. Raises :class:`DataValidationError` with a clear message
    for empty payloads, HTML error pages, malformed CSV or non-numeric content.
    """
    if text is None or not str(text).strip():
        raise DataValidationError(f"{series_id}: empty response (no data returned).")
    head = str(text).lstrip()[:200].lower()
    if head.startswith("<!doctype html") or head.startswith("<html") or "<head" in head:
        raise DataValidationError(
            f"{series_id}: the provider returned an HTML page instead of CSV data "
            "(often a rate-limit, maintenance or proxy page).")
    try:
        raw = pd.read_csv(io.StringIO(text), dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"{series_id}: CSV could not be parsed ({exc}).") from exc
    if raw.shape[1] < 2 or raw.empty:
        raise DataValidationError(f"{series_id}: CSV has no data rows or fewer than two columns.")
    date_col = raw.columns[0]
    value_col = series_id if series_id in raw.columns else raw.columns[1]

    dates = pd.to_datetime(raw[date_col], errors="coerce")
    values = pd.to_numeric(raw[value_col], errors="coerce")
    n_raw = len(raw)
    bad_dates = int(dates.isna().sum())
    ok_date = dates.notna()
    placeholders = int((values.isna() & ok_date).sum())

    s = pd.Series(values[ok_date].to_numpy(dtype=float), index=pd.DatetimeIndex(dates[ok_date]),
                  name=series_id)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        raise DataValidationError(f"{series_id}: no numeric observations after cleaning.")
    n_dups = int(s.index.duplicated(keep="last").sum())
    if n_dups:
        s = s[~s.index.duplicated(keep="last")]
    s = s.sort_index()
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index.name = "date"
    return s, {"n_raw_rows": n_raw, "n_placeholders": placeholders,
               "n_bad_dates": bad_dates, "n_duplicates": n_dups}


def infer_frequency(s: pd.Series) -> str:
    """'daily' if the median gap is < 10 days, otherwise 'monthly'."""
    if len(s) < 3:
        return "monthly"
    gap = pd.Series(s.index).diff().dt.days.median()
    return "daily" if gap < 10 else "monthly"


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

def fetch_fred_csv(series_id: str, timeout: float = C.HTTP_TIMEOUT_S,
                   retries: int = C.HTTP_RETRIES, backoff: float = 1.5,
                   http_get: Optional[Callable] = None,
                   sleep: Callable[[float], None] = time.sleep) -> str:
    """Download one FRED series as CSV text with a timeout and bounded retries.

    ``http_get`` is injectable for tests. Retries on timeouts, connection
    errors, HTTP 429 and 5xx; fails immediately on other 4xx.
    """
    import requests

    getter = http_get or requests.get
    url = C.FRED_CSV_URL.format(series_id=series_id)
    headers = {"User-Agent": "CrudeCortex-GWP2/2.0 (research; +https://github.com/ajeetk095/CrudeCortex-GWP2)"}
    last_err = "unknown error"
    for attempt in range(1, max(1, retries) + 1):
        try:
            resp = getter(url, timeout=timeout, headers=headers)
            status = getattr(resp, "status_code", 200)
            if status == 200:
                text = resp.text
                if not text or not text.strip():
                    last_err = "empty response body"
                else:
                    return text
            elif status in (429,) or 500 <= status < 600:
                last_err = f"HTTP {status}"
            else:
                raise DataUnavailableError(f"{series_id}: provider answered HTTP {status} for {url}.")
        except DataUnavailableError:
            raise
        except Exception as exc:  # requests exceptions, timeouts, proxy errors
            last_err = f"{type(exc).__name__}: {str(exc)[:160]}"
        if attempt < retries:
            sleep(backoff ** attempt)
    raise DataUnavailableError(
        f"{series_id}: FRED could not be reached after {retries} attempt(s) ({last_err}).")


def fetch_yahoo_vix_monthly(start: str = C.START, end: str = C.END) -> pd.Series:
    """Monthly ^VIX close from Yahoo Finance (the report's original method)."""
    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover - optional dependency
        raise DataUnavailableError(f"VIX: yfinance is not installed ({exc}).") from exc
    try:
        raw = yf.download("^VIX", start=start, end=end, interval="1mo",
                          progress=False, auto_adjust=True, timeout=C.HTTP_TIMEOUT_S)
    except Exception as exc:
        raise DataUnavailableError(f"VIX: Yahoo Finance request failed ({type(exc).__name__}).") from exc
    if raw is None or raw.empty or "Close" not in raw:
        raise DataUnavailableError("VIX: Yahoo Finance returned no data.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        raise DataUnavailableError("VIX: Yahoo Finance returned only empty values.")
    close.index = pd.to_datetime(close.index).tz_localize(None).to_period("M").to_timestamp()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.name = "VIXCLS"
    close.index.name = "date"
    return close


# --------------------------------------------------------------------------
# Local files (snapshot + runtime cache)
# --------------------------------------------------------------------------

def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_local(path: str, series_id: str, origin: Origin) -> SeriesLoad:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    s, st = parse_fred_csv_text(text, series_id)
    meta_at = None
    if origin == "cache":
        meta = _read_cache_meta()
        meta_at = meta.get(series_id, {}).get("retrieved_at")
    return SeriesLoad(series_id, s, origin, infer_frequency(s), retrieved_at=meta_at,
                      file_modified_at=_iso(os.path.getmtime(path)), path=path, **st)


def _cache_meta_path(cache_dir: str | None = None) -> str:
    return os.path.join(cache_dir or C.CACHE_DIR, "cache_meta.json")


def _read_cache_meta(cache_dir: str | None = None) -> dict:
    try:
        with open(_cache_meta_path(cache_dir), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _atomic_write(path: str, text: str) -> None:
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_cache(series_id: str, text: str, cache_dir: str | None = None) -> None:
    """Persist a validated download as the 'last successful retrieval'. Never raises."""
    try:
        d = cache_dir or C.CACHE_DIR
        _atomic_write(os.path.join(d, f"{series_id}.csv"), text)
        meta = _read_cache_meta(d)
        meta[series_id] = {"retrieved_at": _now_iso()}
        _atomic_write(_cache_meta_path(d), json.dumps(meta, indent=2))
    except OSError:
        pass  # read-only filesystems must not break the app


def snapshot_path(series_id: str, snapshot_dir: str | None = None) -> str:
    return os.path.join(snapshot_dir or C.SNAPSHOT_DIR, f"{series_id}.csv")


def cache_path(series_id: str, cache_dir: str | None = None) -> str:
    return os.path.join(cache_dir or C.CACHE_DIR, f"{series_id}.csv")


# --------------------------------------------------------------------------
# Public loaders
# --------------------------------------------------------------------------

def load_series(series_id: str, mode: str = "snapshot", *,
                fetcher: Optional[Callable[[str], str]] = None,
                snapshot_dir: str | None = None,
                cache_dir: str | None = None) -> SeriesLoad:
    """Load one FRED series following the provider order documented above."""
    fetcher = fetcher or fetch_fred_csv
    snap, cache = snapshot_path(series_id, snapshot_dir), cache_path(series_id, cache_dir)
    live_error: Optional[str] = None
    live_attempted = False

    def _try_live() -> Optional[SeriesLoad]:
        nonlocal live_error, live_attempted
        live_attempted = True
        try:
            text = fetcher(series_id)
            s, st = parse_fred_csv_text(text, series_id)
        except DataError as exc:
            live_error = str(exc)
            return None
        write_cache(series_id, text, cache_dir)
        return SeriesLoad(series_id, s, "live", infer_frequency(s), retrieved_at=_now_iso(),
                          live_attempted=True, **st)

    def _try_file(path: str, origin: Origin) -> Optional[SeriesLoad]:
        nonlocal live_error
        if not os.path.isfile(path):
            return None
        try:
            ld = _read_local(path, series_id, origin)
        except (DataError, OSError) as exc:
            msg = f"{origin} file unusable: {exc}"
            live_error = f"{live_error} | {msg}" if live_error else msg
            return None
        ld.live_attempted, ld.live_error = live_attempted, live_error
        return ld

    order = (["live", "cache", "snapshot"] if mode == "live" else ["snapshot", "cache", "live"])
    for step in order:
        if step == "live":
            ld = _try_live()
        elif step == "cache":
            ld = _try_file(cache, "cache")
        else:
            ld = _try_file(snap, "snapshot")
        if ld is not None:
            ld.live_attempted, ld.live_error = live_attempted, live_error
            if mode == "live" and ld.origin != "live":
                ld.notes.append("Live FRED retrieval failed; using the most recent local copy.")
            return ld
    raise DataUnavailableError(
        f"{series_id}: no data available. Live FRED failed ({live_error or 'not attempted'}), "
        f"and neither a runtime cache nor the bundled snapshot ({snap}) exists. "
        "Run `bash data/download_fred_data.sh` or `python scripts/refresh_snapshot.py` "
        "from a machine with internet access.")


def load_vix(mode: str = "snapshot", *, fetcher: Optional[Callable[[str], str]] = None,
             yahoo: Optional[Callable[[], pd.Series]] = None,
             snapshot_dir: str | None = None, cache_dir: str | None = None) -> SeriesLoad:
    """VIX: FRED VIXCLS via :func:`load_series`; Yahoo monthly close only as last resort."""
    try:
        return load_series("VIXCLS", mode, fetcher=fetcher, snapshot_dir=snapshot_dir,
                           cache_dir=cache_dir)
    except DataUnavailableError as fred_exc:
        try:
            y = (yahoo or fetch_yahoo_vix_monthly)()
        except DataUnavailableError as y_exc:
            raise DataUnavailableError(
                f"VIX is unavailable from every provider. FRED: {fred_exc} Yahoo: {y_exc}") from y_exc
        return SeriesLoad("VIXCLS", y, "yahoo_fallback", "monthly", retrieved_at=_now_iso(),
                          n_raw_rows=len(y), live_attempted=True, live_error=str(fred_exc),
                          notes=["FRED VIXCLS unavailable; Yahoo Finance monthly close used "
                                 "(the report's original VIX method). Monthly-mean aggregation "
                                 "is not possible from monthly bars."])


def load_all(mode: str = "snapshot", **kw) -> dict[str, SeriesLoad]:
    """Load the four non-VIX FRED series plus VIX; returns ``{series_id: SeriesLoad}``."""
    out: dict[str, SeriesLoad] = {}
    for spec in C.SERIES.values():
        if spec.series_id == "VIXCLS":
            out["VIXCLS"] = load_vix(mode, **kw)
        else:
            out[spec.series_id] = load_series(spec.series_id, mode, **{
                k: v for k, v in kw.items() if k in ("fetcher", "snapshot_dir", "cache_dir")})
    return out
