import streamlit as st
import data_processing
from predict import run_predictions, filter_predictions, pick_11
import threading
import time

# Current gameweek we are predicting for
CURRENT_GAMEWEEK = 4

# Set our data source as remote
data_processing.DATA_SOURCE = data_processing.REMOTE_LOCATION

# Fetch predictions
# Cache data per hour!!!
@st.cache_data(ttl=3600, show_spinner=False)
def get_predictions(gw, model, horizon):
    return run_predictions("2025-2026", gw, model, horizon)

# Fetch optimal_11
# Cache data per hour
@st.cache_data(ttl=3600, show_spinner=False)
def get_optimal_11(gw, model, horizon):
    return pick_11(get_predictions(gw, model, horizon))

# Reruns/refresehes and caches all modelling info/data
def refresh_modelling():
    models = ["V2", "V2_5"] #["V2", "V2_ESI"]
    horizons = [6] #[1, 3, 6, 10]
    gw = CURRENT_GAMEWEEK
    for model in models:
        for h in horizons:
            _ = get_predictions(gw, model, h)
            _ = get_optimal_11(gw, model, h)

# --- Background refresher thread ---
def background_refresher():
    while True:
        refresh_modelling()
        time.sleep(3600)  # refresh every hour

# Kick off background data refreshing thread upon app startup
# Kick off once when app starts (first session)
if "refresher_started" not in st.session_state:
    refresh_modelling()  # immediate prewarm at startup
    threading.Thread(target=background_refresher, daemon=True).start()
    st.session_state.refresher_started = True



### APP ###
st.set_page_config(page_title="Eagle Eye FPL Model")

st.set_page_config(
    page_title="Eagle Eye FPL Model",
    layout="wide"
)

st.title("Eagle Eye FPL Model")

# --- Sidebar - Model Parameters ---
st.sidebar.header("⚙️ Model Parameters")
gameweek = st.sidebar.number_input("Gameweek", 1, 38, CURRENT_GAMEWEEK)
model_name = st.sidebar.selectbox("Model", ["V2", "V2_5", "V2_ESI"], index=0)
horizon = st.sidebar.slider("Horizon (GWs)", 1, 15, 6)

# --- Tabs ---
tab1, tab2 = st.tabs(["🔮 Predictions", "🏆 Optimal 11"])
# Predictions tab
with tab1:
    st.subheader("Run Predictions")
    # Player results filters
    # position
    position_options = ["", "Forward", "Midfielder", "Defender", "Goalkeeper"]
    position = st.selectbox("Position (leave blank for all)", options=position_options, index=0)
    if position == "":
        position = None
    # max_cost
    max_cost = st.slider("Max Cost (£M)", min_value=4.0, max_value=15.0, value=15.0, step=0.5)
    if max_cost == 15.0:
        max_cost = None
    if st.button("Run Predictions"):
        with st.spinner("Running predictions..."):
            predictions = get_predictions(gameweek, model_name, horizon)
        st.success("Predictions ready!")
        # Filter dataframe as desired
        predictions = filter_predictions(predictions, position, max_cost)
        st.dataframe(predictions)
# Optimal 11 tab
with tab2:
    st.subheader("Generate Optimal XI")
    if st.button("Generate Optimal XI"):
        with st.spinner("Finding Optimal 11..."):
            optimal_11 = get_optimal_11(gameweek, model_name, horizon)
        st.success("Optimal XI ready!")
        st.dataframe(optimal_11)
