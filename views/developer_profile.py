"""Developer Profile — Ajeet Kumar."""
import os

import streamlit as st

from core import config as C
from core import ui_components as ui

D = C.DEVELOPER
ui.page_header("Developer Profile", "The person behind CrudeCortex.", "Results & documentation")

photo = next((os.path.join(C.ASSETS_DIR, p) for p in D["photo_candidates"] if os.path.isfile(os.path.join(C.ASSETS_DIR, p))), None)
c1, c2 = st.columns([1, 3])
with c1:
    if photo:
        st.image(photo, width="stretch")
    else:
        st.markdown('<div style="aspect-ratio:1;border-radius:16px;background:linear-gradient(135deg,#12303A,#0A0D10);border:1px solid #E8A33D55;'
                    'display:flex;align-items:center;justify-content:center;font-size:3rem;color:#E8A33D;font-weight:600;">AK</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f"## {D['name']}")
    st.markdown(D["positioning"])
    st.link_button("GitHub", D["github"])
    st.link_button("LinkedIn", D["linkedin"])
if not photo:
    st.caption("To show a photo, add `assets/ajeet_kumar.jpg` (or .png/.webp) to the repository and redeploy.")

ui.section("Education")
st.markdown("- B.Tech, Information Technology — Dr. A.P.J. Abdul Kalam Technical University\n"
            "- M.Tech, Computer Science Engineering — NSUT Delhi\n"
            "- MSc, Financial Engineering — WorldQuant University")
ui.section("Experience & recognition")
st.markdown("- Internship, Algo8.ai — AI-driven defect detection with computer vision\n- Internship, Amul GCMMF\n"
            "- 1st place, Algo8 ML Hackathon at IIT Kanpur\n- Felicitated by the President of India for academic excellence")
ui.section("Skills")
st.markdown("**Languages:** Python, C++, SQL  \n**Quant methods:** Markowitz MVO, Black–Litterman, Monte Carlo simulation, CVaR, Ledoit–Wolf shrinkage  \n"
            "**ML / data:** NumPy, pandas, SciPy, scikit-learn, statsmodels, PyTorch, XGBoost, LightGBM, OpenCV, FastAPI, Tableau, Git")
ui.section("Role in this project")
st.markdown("Application engineering and deployment of the CrudeCortex GWP2 research (MScFE 660, Group 14660): data pipeline, model engines, interactive Streamlit interface, testing and documentation.")
ui.disclaimer()
