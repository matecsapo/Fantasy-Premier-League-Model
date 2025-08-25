import argparse
import joblib
import pandas as pd
import numpy as np
from features import get_data
from model_V2 import MODEL_V2_FEATURES
from model_V2_ESI import MODEL_V2_ESI_FEATURES
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from tabulate import tabulate
import pulp

# Predicts FPL performance for upcoming [horizon] gws according to specified parameters
def predict(season, gameweek, model_name, horizon):
    # Load desired model
    model = joblib.load(f"models/{model_name}.pkl")
    chosen_model_feature_set = []
    if model_name == "V2":
        chosen_model_feature_set = MODEL_V2_FEATURES
    elif model_name == "V2_ESI":
        chosen_model_feature_set = MODEL_V2_ESI_FEATURES
    # List of players + identifying info to predict for (all)
    predictions = get_data(season, gameweek, gameweek, 0)
    predictions = predictions[["player_id", "first_name", "second_name", "position", "team_name", "now_cost"]]
    # Predict players' performance across next [horizon] gws
    for game in range(0, horizon):
        features = get_data(season, gameweek, gameweek, game)
        #features.to_csv(f"game_{game + 1}.csv")
        predictions[f"opponent_game_{game + 1}"] = features["opponent_name"]
        model_features = features[chosen_model_feature_set]
        predictions[f"predicted_points_game_{game + 1}"] = model.predict(model_features)
    return predictions   

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

# Picks optimal starting 11 across [horizon] game forecast
def pick_11(predictions):
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
    ]) <= 83.0
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
    optimal_11 = optimal_11[['first_name', 'second_name', 'position', 'team_name', 'now_cost', 'linear_horizon_performance_score', "predicted_horizon_points"]]
    # order by position, lin_hor_perf_score
    optimal_11['position'] = pd.Categorical(optimal_11['position'], categories=["Goalkeeper", "Defender", "Midfielder", "Forward"], ordered=True)
    optimal_11 = optimal_11.sort_values(by=["position", "linear_horizon_performance_score"], ascending=False)
    return optimal_11

# Displays all generated predictions
def display_predictions(predictions, optimal_11, horizon):
    # Order by best-to-worst expected performers over [horizon] upcoming games
    predictions = predictions.sort_values(by="linear_horizon_performance_score", ascending=False)

    # Save all predictions data to file
    horizon_cols = []
    for i in range(1, horizon + 1):
        horizon_cols.append(f"opponent_game_{i}")
        horizon_cols.append(f"predicted_points_game_{i}")
    file_info = ["player_id", "first_name", "second_name", "position", "team_name", "now_cost", "predicted_horizon_points", "linear_horizon_performance_score"] \
                + horizon_cols
    file_data = predictions[file_info]
    file_data.to_csv("predictions.csv")

    # Display most important data to terminal
    #terminal_info = ["second_name", "position", "team_name", "now_cost", "predicted_horizon_points", "linear_horizon_performance_score", "opponent_game_1"]
    #terminal_data = predictions[terminal_info]
    #data_to_print = predictions[terminal_info].head(100).astype(str)
    #print(f"HORIZON = {horizon} POINTS PREDICTIONS")
    #print(tabulate(data_to_print.values, headers=data_to_print.columns, tablefmt="plain", stralign="left", numalign="left"))

    # Display optimal 11 - both file and terminal
    optimal_11.to_csv("optimal_11.csv")
    #print("OPTIMAL 11 : ")
    #data_to_print = optimal_11.astype(str)
    #print(tabulate(data_to_print.values, headers=data_to_print.columns, tablefmt="plain", stralign="left", numalign="left"))
    #print()
    #print(f"Total Cost = {optimal_11["now_cost"].sum():.1f}")
    #print(f"Total lin_hor_perf_score = {optimal_11["linear_horizon_performance_score"].sum():.1f}")
    #print(f"Total pred_horizon_points = {optimal_11["predicted_horizon_points"].sum():.1f}")

# Predicts FPL performance according to specified parameters
def main():
    # command-line arguments specifying behaviour
    parser = argparse.ArgumentParser(description="FPL Prediction Engine")
    parser.add_argument("--season", type=str, default="2025-2026", help="season predicting for")
    parser.add_argument("--gameweek", type=int, required=True, help="next/upcoming (yet unstarted) gameweek")
    parser.add_argument("--model", type=str, default="V2", help="model to use")
    parser.add_argument("--horizon", type=int, default=6, help="horizon of gws into future to consider when predicting")
    args = parser.parse_args()

    # Predict performance accross upcoming [horizon] gws according to specified parameters
    predictions = predict(args.season, args.gameweek, args.model, args.horizon)

    # Add performance calculations/metrics
    predictions = add_calculations(predictions, args.horizon)

    # Pick best 11
    optimal_11 = pick_11(predictions)
    
    # Output all determined info
    display_predictions(predictions, optimal_11, args.horizon)

if __name__ == "__main__":
    main()