# CrudeCortex — GWP2
*The Brain Behind the Barrel.*

Interactive research application for **MScFE 660 Risk Management · Group Work Project 2** (Group 14660): WTI crude-oil regime detection with a **3-state Gaussian Hidden Markov Model** and a **Bayesian Network** relating the regimes to VIX, industrial production, energy CPI and the US dollar index.

> Descriptive analysis of historical monthly data. Not a forecast, trading signal or investment advice. Bayesian-network edges are conditional associations, not causal effects.

## What it contains
| Group | Pages |
|---|---|
| Research Overview | Home · Research Objectives · Data & Indicators |
| HMM Research | HMM Algorithms Lab · Market Regime Identification · HMM Definition & Design · Regime Detection Pipeline |
| Bayesian Research | Bayesian Network Lab · Conditional Probability Explorer |
| Results & Documentation | Summary Dashboard · Methodology & Limitations · Documentation · Developer Profile |

Methodology preserved from the submitted work: four toy-HMM algorithms, rule-based Bull/Bear/Stagnant benchmark, Gaussian HMM (full covariance, 200 iterations, tol 1e-4, 10 restarts, best converged log-likelihood), five-node Bayesian Network (Hill Climb, BIC, quantile discretisation, CPDs, exact inference). With the default settings the app reproduces the report's printed ADF statistic, HMM log-likelihood, regime counts, Table 14 and learned edge (checked by tests and shown on the Summary Dashboard).

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
No API key, database or paid service is needed. Data come from the bundled FRED CSV snapshot (`data/fred/`); "Refresh data from FRED" optionally pulls the public CSV endpoint and falls back safely if it is unreachable.

## Tests
```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

## Layout
`app.py` navigation · `views/` pages · `core/` engines (`data_sources`, `data_gwp2`, `hmm_engine`, `bn_engine`), charts, exports, state · `data/fred/` snapshot + `MANIFEST.json` · `documents/`, `notebooks/` original academic material (unchanged) · `docs/` registers and reports · `scripts/refresh_snapshot.py`.

See `USER_GUIDE.md`, `DEPLOYMENT.md`, `CHANGELOG.md`, `docs/DATA_AND_DISCREPANCY_REGISTER.md`, `docs/UNRESOLVED_LIMITATIONS.md`, `docs/VERIFICATION_REPORT.md`.

Developer: Ajeet Kumar — [GitHub](https://github.com/ajeetk095) · [LinkedIn](https://www.linkedin.com/in/ajeek095/). Licence: see `LICENSE`.
