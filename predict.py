import argparse
import joblib
import pandas as pd
import numpy as np
import shap
import os
from features import get_data
from model_V2 import MODEL_V2_FEATURES
from model_V2_ESI import MODEL_V2_ESI_FEATURES
from model_V2_5 import MODEL_V2_5_FEATURES
from model_V3 import MODEL_V3_FEATURES
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pulp

# Predicts FPL performance for upcoming [horizon] gws according to specified parameters
def predict(season, gameweek, model_name, horizon):
    # Load desired model
    model = joblib.load(f"models/{model_name}/{model_name}.pkl")
    chosen_model_feature_set = []
    if model_name == "V2":
        chosen_model_feature_set = MODEL_V2_FEATURES
    elif model_name == "V2_ESI":
        chosen_model_feature_set = MODEL_V2_ESI_FEATURES
    elif model_name == "V2_5":
        chosen_model_feature_set = MODEL_V2_5_FEATURES
    elif model_name == "V3":
        chosen_model_feature_set = MODEL_V3_FEATURES
    # List of players + identifying info to predict for (all)
    predictions = get_data(season, gameweek, gameweek, 0, True) # This was adjusted from False to True
    predictions = predictions[["player_id", "first_name", "second_name", "position", "team_name", "now_cost"]]
    # Filter out blacklisted players
    with open("blacklist.txt", "r") as f:
        blacklist = [line.strip() for line in f.readlines()]
    predictions = predictions[~predictions["second_name"].isin(blacklist)].reset_index(drop=True)
    # Collect shap values
    explainer = shap.TreeExplainer(model)
    shap_explanations = pd.DataFrame(columns=["player_id", "game"] + chosen_model_feature_set)   
    # Predict players' performance across next [horizon] gws
    for game in range(0, horizon):
        features = get_data(season, gameweek, gameweek, game, True) # This was adjusted from False to True
        # Remove blacklisted players
        features = features[~features["second_name"].isin(blacklist)].reset_index(drop=True)
        predictions[f"opponent_game_{game + 1}"] = features["opponent_name"]
        model_features = features[chosen_model_feature_set]
        # Generate predictions
        predictions[f"predicted_points_game_{game + 1}"] = model.predict(model_features)
        # Retrieve predictions' shap values
        shap_values = explainer.shap_values(model_features)
        shap_values = pd.DataFrame(shap_values, columns=chosen_model_feature_set)
        shap_values.insert(0, "player_id", features["player_id"])
        shap_values.insert(0, "game", game + 1)
        shap_explanations = pd.concat([shap_explanations, shap_values], ignore_index=True)
    return predictions, shap_explanations  

# Adds performance related calculations to predictions
def add_calculations(predictions, horizon):
    gw_cols = [f"predicted_points_game_{i+1}" for i in range(horizon)]
    # Total predicted points across all [horizon] games
    predictions["predicted_horizon_points"] = predictions[gw_cols].sum(axis=1)
    
    # Linearly-decay-weighted performance score over [horizon] games
    # Example --> [horizon] == 6 --> weights = [6, 5, 4, 3, 2, 1]
    weights = np.arange(horizon, 0, -1)
    weights = weights / weights.sum()
    weighted_gw_points = predictions[gw_cols].multiply(weights, axis=1)
    predictions["linear_horizon_performance_score"] = weighted_gw_points.sum(axis=1)
    return predictions

# Formats predictions
def format_predictions(predictions, horizon):
    # Order by best-to-worst expected performers over [horizon] upcoming games
    predictions = predictions.sort_values(by="linear_horizon_performance_score", ascending=False)
    predictions = predictions.reset_index(drop=True)
    # Specificy column presence + order
    horizon_cols = []
    for i in range(1, horizon + 1):
        horizon_cols.append(f"opponent_game_{i}")
        horizon_cols.append(f"predicted_points_game_{i}")
    info = ["player_id", "first_name", "second_name", "position", "team_name", "now_cost", "linear_horizon_performance_score", "predicted_horizon_points"] \
                + horizon_cols
    predictions = predictions[info]
    return predictions

# Generates model predictions for given parameters
def run_predictions(season, gameweek, model, horizon):
    # Predict performance accross upcoming [horizon] gws according to specified parameters
    predictions, shap_explanations = predict(season, gameweek, model, horizon)

    # Add performance calculations/metrics
    predictions = add_calculations(predictions, horizon)

    # Format data
    predictions = format_predictions(predictions, horizon)

    # Return generated predictions
    return predictions, shap_explanations

# Filter predictions
def filter_predictions(predictions, position, max_cost):
    if position != None:
        predictions = predictions[predictions["position"] == position]
    if max_cost != None:
        predictions = predictions[predictions["now_cost"] <= max_cost]
    # Reset index for easy visual ordering/ranking viewing
    predictions = predictions.reset_index(drop=True)
    return predictions

# Picks optimal starting 11 across [horizon] game forecast
def pick_11(predictions):
    predictions = predictions.dropna(subset=['now_cost'])
    # set up a target value maximization problem
    prob = pulp.LpProblem("FPL_11_Optimization", pulp.LpMaximize)
    players = predictions['player_id'].tolist()
    # Decision variables --> 1 = player in squad, 0 = not
    x = pulp.LpVariable.dicts("player", players, cat='Binary')
    # Set objective function --> maximize total linear_horizon_performance_score
    prob += pulp.lpSum([
        predictions.loc[predictions['player_id']==pid, 'linear_horizon_performance_score'].values[0] * x[pid] 
        for pid in players
    ])
    # Specify squad selection constraints
    # Pick a starting 11 = 11 players
    prob += pulp.lpSum([x[pid] for pid in players]) == 11
    # Position constraints
    pos_limits = {'Goalkeeper': (1,1), 'Defender': (3,5), 'Midfielder': (3,5), 'Forward': (1,3)}
    for pos, (min_p, max_p) in pos_limits.items():
        pos_players = predictions[predictions['position']==pos]['player_id'].tolist()
        prob += pulp.lpSum([x[pid] for pid in pos_players]) >= min_p
        prob += pulp.lpSum([x[pid] for pid in pos_players]) <= max_p
    # Budget constraint --> 83M... 17M = min to create bench!
    prob += pulp.lpSum([
        predictions.loc[predictions['player_id']==pid, 'now_cost'].values[0] * x[pid] 
        for pid in players
    ]) <= 83.5
    # Max 3 players per team
    teams = predictions['team_name'].unique()
    for team in teams:
        team_players = predictions[predictions['team_name']==team]['player_id'].tolist()
        prob += pulp.lpSum([x[pid] for pid in team_players]) <= 3
    # Solve the LP problem
    prob.solve((pulp.PULP_CBC_CMD(msg=False)))
    # return players in optimal 11
    selected_ids = [pid for pid, var in x.items() if pulp.value(var) == 1]
    optimal_11 = predictions[predictions['player_id'].isin(selected_ids)].copy()
    # Format optimal_11
    optimal_11 = optimal_11[['first_name', 'second_name', 'position', 'team_name', 'now_cost', 'linear_horizon_performance_score', "predicted_horizon_points", "opponent_game_1", "predicted_points_game_1"]]
    # order by position, lin_hor_perf_score
    optimal_11 = optimal_11.copy()
    optimal_11['position'] = pd.Categorical(optimal_11['position'], categories=["Goalkeeper", "Defender", "Midfielder", "Forward"], ordered=True)
    optimal_11 = optimal_11.sort_values(by=["position", "linear_horizon_performance_score"], ascending=False)
    optimal_11 = optimal_11.reset_index(drop=True)
    # Add critical metrics
    optimal_11.loc[len(optimal_11)] = {}
    optimal_11.loc[len(optimal_11)] = {
        "second_name": "Total",
        "now_cost": round(optimal_11["now_cost"].sum(), 1),
        "linear_horizon_performance_score": round(optimal_11["linear_horizon_performance_score"].sum(), 1),
        "predicted_horizon_points": round(optimal_11["predicted_horizon_points"].sum(), 1)
    }
    return optimal_11

# Displays all generated predictions
def display_predictions(predictions, shap_values, position, max_cost, optimal_11, model_name, gameweek):
    # Directory for storying predictions
    os.makedirs(f"predictions/GW_{gameweek}/model_{model_name}", exist_ok=True)
    # Filter to target position +/ price if specified
    predictions = filter_predictions(predictions, position, max_cost)
    # Save predictions to file
    predictions.to_csv(f"predictions/GW_{gameweek}/model_{model_name}/predictions.csv")
    
    # Save shap values to file
    shap_values.to_csv(f"predictions/GW_{gameweek}/model_{model_name}/shap_explanations.csv")

    # Display optimal 11 to file
    optimal_11.to_csv(f"predictions/GW_{gameweek}/model_{model_name}/optimal_11.csv")

if __name__ == "__main__":
    # command-line arguments specifying behaviour
    parser = argparse.ArgumentParser(description="FPL Prediction Engine")
    parser.add_argument("--season", type=str, default="2025-2026", help="season predicting for")
    parser.add_argument("--gameweek", type=int, required=True, help="next/upcoming (yet unstarted) gameweek")
    parser.add_argument("--model", type=str, default="V2", help="model to use")
    parser.add_argument("--horizon", type=int, default=6, help="horizon of gws into future to consider when predicting")
    parser.add_argument("--position", type=str, help="Filters to only show predictions for given position")
    parser.add_argument("--max_cost", type=float, help="Filters to only show predictions <= given cost")
    args = parser.parse_args()

    # Run predictions
    predictions, shap_values = run_predictions(args.season, args.gameweek, args.model, args.horizon)

    # Get Optimal 11
    optimal_11 = pick_11(predictions)

    # Display predictions
    display_predictions(predictions, shap_values, args.position, args.max_cost, optimal_11, args.model, args.gameweek)

    