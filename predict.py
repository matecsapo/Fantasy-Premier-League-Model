import pandas as pd
import joblib
import os

NUM_GWS_TO_ROLL = 3

def attach_ema_features(players_df, playerstats_csv="2025-2026/playerstats.csv"):
    """
    Attach EMA features from GW1 playerstats to players_df.
    Builds exactly the features your model expects, with prefix ema{NUM_GWS_TO_ROLL}_.
    """
    # Load GW1 playerstats only
    playerstats = pd.read_csv(playerstats_csv)
    playerstats = playerstats[playerstats["gw"] == 1]

    # Minutes buckets
    minutes = playerstats["minutes"].fillna(0)
    playerstats["minutes_bucket_Low"]    = (minutes <= 30).astype(int)
    playerstats["minutes_bucket_Medium"] = ((minutes > 30) & (minutes <= 60)).astype(int)
    playerstats["minutes_bucket_High"]   = ((minutes > 60) & (minutes < 90)).astype(int)
    playerstats["minutes_bucket_Ironman"] = (minutes == 90).astype(int)

    # Mapping of columns to EMA feature names
    feature_map = {
        "minutes": f"ema{NUM_GWS_TO_ROLL}_minutes",
        "minutes_bucket_Low": f"ema_minutes_bucket_0-30",
        "minutes_bucket_Medium": f"ema_minutes_bucket_30-60",
        "minutes_bucket_High": f"ema_minutes_bucket_60-90",
        "minutes_bucket_Ironman": f"ema_minutes_bucket_90+",
        "expected_assists": f"ema{NUM_GWS_TO_ROLL}_expected_assists",
        "expected_goals": f"ema{NUM_GWS_TO_ROLL}_expected_goals",
        "expected_goal_involvements": f"ema{NUM_GWS_TO_ROLL}_expected_goal_involvements",
        "expected_goals_conceded": f"ema{NUM_GWS_TO_ROLL}_expected_goals_conceded",
        "clean_sheets": f"ema{NUM_GWS_TO_ROLL}_clean_sheets",
        "saves": f"ema{NUM_GWS_TO_ROLL}_saves",
        "red_cards": f"ema{NUM_GWS_TO_ROLL}_red_cards",
        "total_points": f"ema{NUM_GWS_TO_ROLL}_total_points",
        "ict_index": f"ema{NUM_GWS_TO_ROLL}_ict_index",
        "influence": f"ema{NUM_GWS_TO_ROLL}_influence",
        "now_cost": f"ema{NUM_GWS_TO_ROLL}_value",
        "selected_by_percent": f"ema{NUM_GWS_TO_ROLL}_selected"
    }

    # Create new DataFrame with mapped EMA feature names
    ema_features = pd.DataFrame()
    ema_features["player_id"] = playerstats["id"]
    ema_features["first_name"] = playerstats["first_name"]
    ema_features["second_name"] = playerstats["second_name"]

    for orig_col, new_col in feature_map.items():
        ema_features[new_col] = playerstats[orig_col]

    return players_df.merge(ema_features, on="player_id", how="inner")

def attach_team_strength(players_df, teams_csv="2025-2026/teams.csv"):
    """
    Adds 'team_strength' to each player in players_df based on their 'team_code'.
    """
    teams = pd.read_csv(teams_csv)
    
    # Only need mapping: team_code -> strength
    team_strength_map = teams.set_index("code")["strength"].to_dict()
    
    # Map each player's team_code to strength
    players_df["team_strength"] = players_df["team_code"].map(team_strength_map)
    
    return players_df


def attach_gw_fixture_info(players_df, gw_fixtures_csv="2025-2026/By Gameweek/GW2/fixtures.csv", teams_csv="2025-2026/teams.csv"):
    """
    Attaches GW2 fixture info (is_home, opponent_strength, strength_diff) to players_df.
    """
    # Load fixtures and teams
    fixtures = pd.read_csv(gw_fixtures_csv)
    teams = pd.read_csv(teams_csv)

    # Map team_code -> strength
    team_strength_map = teams.set_index("code")["strength"].to_dict()

    # Prepare fixture info: determine for each team who their opponent is and if home
    fixture_rows = []
    for _, row in fixtures.iterrows():
        # Home team
        fixture_rows.append({
            "team_code": row["home_team"],
            "is_home": 1,
            "opponent_code": row["away_team"]
        })
        # Away team
        fixture_rows.append({
            "team_code": row["away_team"],
            "is_home": 0,
            "opponent_code": row["home_team"]
        })

    fixtures_expanded = pd.DataFrame(fixture_rows)

    # Map opponent strength
    fixtures_expanded["opponent_strength"] = fixtures_expanded["opponent_code"].map(team_strength_map)

    # Merge into players
    players_df = players_df.merge(
        fixtures_expanded[["team_code", "is_home", "opponent_strength"]],
        on="team_code",
        how="left"
    )

    # Add strength difference
    players_df["strength_diff"] = players_df["team_strength"] - players_df["opponent_strength"]

    return players_df

# Usage
players = pd.read_csv("2025-2026/players.csv")
players = attach_ema_features(players)
players = attach_team_strength(players)
players = attach_gw_fixture_info(players)

pos_map = {
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD"
}

pred_list = []

for pos_full, pos_suffix in pos_map.items():
    model_path = f"models/test.json_{pos_suffix}"
    if not os.path.exists(model_path):
        continue
    
    model = joblib.load(model_path)
    subset = players[players["position"] == pos_full].copy()
    if subset.empty:
        continue
    
    # Keep metadata
    meta = subset[["player_id", "position"]].copy()

    # Read players.csv to get names
    players_file = "2025-2026/players.csv"
    players_full = pd.read_csv(players_file)[["player_id", "second_name"]]

    # Merge to attach names
    meta = meta.merge(
        players_full,
        on="player_id",
        how="left"
    )
    
    # Only pass feature columns to model
    feature_cols = model.feature_names_in_  # RandomForest sets this after training
    model_input = subset[feature_cols]
    
    # Predict points
    meta["predicted_points"] = model.predict(model_input)
    pred_list.append(meta)

predictions_df = pd.concat(pred_list, ignore_index=True)
predictions_df = predictions_df.sort_values(by="predicted_points", ascending=False)
predictions_df = predictions_df[predictions_df["position"] == "Defender"]
# Show all columns and all rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(predictions_df.head(50))