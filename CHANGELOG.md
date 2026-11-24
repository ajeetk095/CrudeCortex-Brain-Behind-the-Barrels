# Changelog
## 2.0.0 — overhaul
- New navigation (4 groups, 13 pages) in `views/`; legacy `pages/` removed (originals remain in the source ZIP).
- Data: provider chain with timeouts/retries/cache/validation, provenance table, window derived from data (D1), VIX handling (D2/D3).
- HMM: log-space forward/backward/Viterbi; fixed Baum–Welch (log-likelihood history, monotone); restart table; independent verification vs hmmlearn; state-order consistency.
- BN: fixed Hill Climb parent handling; AIC/BIC no longer scaled by T; tie-tolerant search; exact inference with error handling; hold-out check; variant comparison; pgmpy cross-check (dev only).
- Interactive controls with explicit Run/Reset; deviation banners; CSV/ZIP exports.
- Tests: 13 original + 30 new. Docs and deployment guide added.
