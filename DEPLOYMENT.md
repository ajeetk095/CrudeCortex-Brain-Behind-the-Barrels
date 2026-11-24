# Deploying to Streamlit Community Cloud (free)
Deployment has **not** been performed or confirmed by the author of this package; follow these steps and verify the live URL yourself.
1. Push this folder to GitHub (`ajeetk095/CrudeCortex-GWP2`), including `data/fred/*.csv` and `data/fred/MANIFEST.json` (the app must run offline).
2. Do **not** commit `.env`, `.streamlit/secrets.toml` or `data/cache/`.
3. At share.streamlit.io choose *Create app* → repository, branch `main`, main file `app.py`.
4. Advanced settings → Python **3.12** (matches `.python-version`; tested locally on 3.13 only).
5. No secrets are required. Optional env vars: `HTTP_TIMEOUT_S`, `HTTP_RETRIES`, `CACHE_DIR`.
6. Deploy; first start installs `requirements.txt` (several minutes).
7. If the build fails on a dependency, copy the log's failing package and relax its upper bound in `requirements.txt`.
8. Smoke-test: open every page; press *Run analysis* on each control group; download the ZIP.
9. Free apps sleep when idle; the first visit afterwards re-runs the cached model fit (seconds).
10. To update data: run `python scripts/refresh_snapshot.py` on a machine with internet, commit the CSVs and manifest, and push.
Custom subdomain (e.g. `crudecortex295`) is set in the app's settings on the platform.
