import streamlit as st
import altair as alt
import data_processing
import pandas as pd
from predict import run_predictions, filter_predictions, pick_11
import threading
import time

# Current gameweek we are predicting for
CURRENT_GAMEWEEK = 8

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
    predictions, _ = get_predictions(gw, model, horizon)
    return pick_11(predictions)

# Reruns/refresehes and caches all modelling info/data
def refresh_modelling():
    models = ["V2", "V2_5"]
    horizons = [6] #[1, 3, 6, 10]
    gw = CURRENT_GAMEWEEK
    for model in models:
        for h in horizons:
            _, _ = get_predictions(gw, model, h)
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
tab1, tab2, tab3 = st.tabs(["🔮 Predictions", "🏆 Optimal 11", "📊 SHAP Values"])
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
            predictions, _ = get_predictions(gameweek, model_name, horizon)
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
# Shap explanations tab
with tab3:
    st.subheader("SHAP Value Prediction Explanations")
    # Select desired player + game
    predictions, shap_values = get_predictions(gameweek, model_name, horizon)
    players = predictions["second_name"]
    selected_player = st.selectbox("Player", [""] + sorted(players))
    gws = range(1, gameweek + 1)
    selected_gw = st.selectbox("Game", gws)
    if selected_player and st.button("Plot SHAP Contributions"):
        # Generate SHAP contributions plot
        selected_player_id = predictions.loc[predictions["second_name"] == selected_player, "player_id"].values[0]
        selected_shap = shap_values.loc[(shap_values["player_id"] == selected_player_id) & (shap_values["game"] == selected_gw)]
        selected_shap = selected_shap.select_dtypes(include="number")
        selected_shap = selected_shap.iloc[0]
        top_features = selected_shap.abs().sort_values(ascending=False).head(10).index
        selected_shap = selected_shap[top_features]
        selected_shap = pd.DataFrame({
            "feature": selected_shap.index,
            "shap_value": selected_shap.values
        })
        chart = alt.Chart(selected_shap).mark_bar().encode(
            y=alt.Y(
                'feature',
                sort=selected_shap.reindex(selected_shap.shap_value.abs().sort_values(ascending=False).index)['feature'].tolist()
            ),
            x='shap_value',
            color=alt.condition(
                alt.datum.shap_value > 0,
                alt.value("steelblue"),
                alt.value("salmon")
            )
        ).properties(
            width=600,
            height=400
        )
        st.altair_chart(chart)