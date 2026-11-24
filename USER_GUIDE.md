# User guide
1. **Start at Home**, then follow the sidebar top to bottom.
2. **Change settings** in the expanders on *Data & Indicators*, *Regime Detection Pipeline* and *Bayesian Network Lab*. Nothing recomputes until you press **Run analysis**. **Reset** restores the research defaults. A banner marks any run that differs from the submitted report.
3. **Dates**: choose the first and last *modelled* month. The app adds a 12-month warm-up automatically and reports the effective window.
4. **VIX aggregation**: month-end close (report default) or monthly mean.
5. **HMM**: states (2–5), restarts, iterations, tolerance, covariance type. Non-convergence is reported, never hidden.
6. **Bayesian network**: max parents, BIC/AIC, discretisation, variable definition, indicators.
7. **Conditional Probability Explorer**: pick evidence, read exact-network and empirical probabilities; few matching months triggers a warning.
8. **Exports**: CSV buttons on each page; *Summary Dashboard* offers a ZIP with all tables, the configuration JSON and a text summary.
9. **Refresh data**: the Data page can request live FRED retrieval; failures fall back to cache or snapshot and the provenance table says which was used.
Every concept has a *plain-language* and a *research-detail* explanation.
