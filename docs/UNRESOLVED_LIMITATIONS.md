# Unresolved limitations
- Live FRED and Yahoo retrieval could not be exercised in the build sandbox (network blocked); only mocked-provider tests cover it.
- Snapshot provenance (retrieval date) unknown; FRED history is revised over time.
- Only tested on Python 3.13 with the versions in the verification report; lower bounds in `requirements.txt` and Python 3.12 are untested.
- Google Fonts import is blocked in the sandbox; the UI falls back to system fonts there.
- Not deployed by the author of this package; the live URL is unverified.
- No developer photo in the repository: a placeholder is shown (add `assets/ajeet_kumar.jpg`).
- Statistical limits (small sample, Gaussian emissions, hindsight labels, same-sample structure learning, edge-direction non-identifiability) are inherent to the method; see Methodology page.
