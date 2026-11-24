# Data & discrepancy register
Same content as the in-app Methodology page.
| ID | Topic | Resolution |
|---|---|---|
| D1 | Report says "Jan 2006"/2005 start; DTWEXBGS begins 2006-01-01 → panel 2006-01…2024-12 (228 rows), model months 2007-01…2024-12 (216) after 12-month warm-up | Window derived from data and displayed |
| D2 | VIX FRED VIXCLS vs Yahoo ^VIX | FRED primary; Yahoo last resort, flagged |
| D3 | VIX monthly aggregation | Default month-end close (reproduces report); monthly mean selectable |
| D4 | Discretisation 33/67 percentile vs qcut | Both offered; default report |
| D5 | BN variables: report text (levels) vs executed notebook (VIX level + trends) | Default notebook; level variant selectable |
| D6 | Score-equivalent edge orientation | Convention: links into WTI_Regime |
| D7 | Bundled CSVs: retrieval date unknown | Checksums in `data/fred/MANIFEST.json`; refresh script |
Note: the snapshot extends to 2026-10; the default window stops at 2024-12 as in the report.
