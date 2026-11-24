# Verification report
Environment: Linux sandbox, Python 3.13.18, streamlit 1.67.0, pandas 3.0.5, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, statsmodels 0.15.0, hmmlearn 0.3.3, plotly 7.1.0.

| Check | Result |
|---|---|
| Original 13 tests | pass (before and after the overhaul) |
| Full suite (`pytest tests`) | 44 passed |
| All 13 pages under Streamlit AppTest | no exceptions |
| Real browser (Chromium/Playwright), click-through of all 13 pages at 1440 px | no Streamlit exceptions |
| Mobile width 390 px (home) | no horizontal scroll |
| Report reproduction (default config) | ADF −9.7807, 228×5 panel, 216×2 features, HMM log-likelihood −305.977, regime counts 129/59/28, Table 14, single BN edge VIX_Level→WTI_Regime |
| Own forward/Viterbi vs hmmlearn | log-likelihood gap ~1e-13, Viterbi agreement 100% |
| BN engine vs pgmpy (dev-only, 16 configurations) | identical scores |
| Provider failure modes (mocked) | live failure → cache/snapshot; corrupt/missing files → clear errors; Yahoo VIX last resort |

## Not verified
- Live FRED/Yahoo retrieval (sandbox network blocked) — mocked only.
- Python 3.12 and the minimum dependency versions.
- Deployment on Streamlit Community Cloud — not performed; no live URL confirmed.
- External links (GitHub/LinkedIn) were not opened from the sandbox.
- Google Fonts could not load in the sandbox (system-font fallback was what was screenshotted).
- Interactive control clicks were covered by AppTest only for the presence of Run buttons; the apply/reset flows were not exercised end-to-end.
