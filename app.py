import streamlit as st
import data_processing
from predict import run_predictions

# Set our data source as remote
data_processing.DATA_SOURCE = data_processing.REMOTE_LOCATION

st.set_page_config(page_title="Eagle Eye FPL Model")

st.set_page_config(
    page_title="Eagle Eye FPL Model",
    layout="wide"
)

st.title("Eagle Eye FPL Model")

# Arguments/parametere specification controls
col1, col2, col3 = st.columns(3)
with col1:
    gameweek = st.slider("Gameweek", min_value=1, max_value=38, value=3)
    position_options = ["", "Forward", "Midfielder", "Defender", "Goalkeeper"]
    position = st.selectbox("Position (leave blank for all)", options=position_options, index=0)
    if position == "":
        position = None
    

with col2:
    model_name = st.selectbox("Select Model", options=["V2", "V2_ESI"], index=0)
    max_cost = st.slider("Max Cost (£M, optional)", min_value=4.0, max_value=15.0, value=15.0, step=0.5)
    if max_cost == 15.0:
        max_cost = None

with col3:
    horizon = st.slider("Prediction Horizon (weeks)", min_value=1, max_value=20, value=6)

# Run predictions
# Cache data per hour!!!
@st.cache_data(ttl=3600)
def get_predictions(gw, model, horizon, position, max_cost):
    return run_predictions(
        season="2025-2026",
        gameweek=gw,
        model=model,
        horizon=horizon,
        position=position,
        max_cost=max_cost
    )

# ---- Run button ----
if st.button("Run Predictions"):
    predictions, optimal_11 = get_predictions(gameweek, model_name, horizon, position, max_cost)
    # Display side by side
    st.info("💡 Tip: Click the 'Expand' icon at the top-right of each table to view it fullscreen.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Optimal 11")
        st.dataframe(optimal_11)
    with col2:
        st.markdown("### Player Predictions")
        st.dataframe(predictions)
