import pandas as pd
import joblib
import os
import glob
import unicodedata
from functools import reduce

NUM_GWS_TO_ROLL = 1

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


def attach_gw_fixture_info(players_df, gw, base_path="2025-2026/By Gameweek", teams_csv="2025-2026/teams.csv"):
    """
    Attaches fixture info for a specific gameweek `gw` to players_df.
    Adds columns: is_home, opponent_strength, strength_diff.
    """
    import os

    # Construct the GW fixtures path dynamically
    gw_fixtures_csv = os.path.join(base_path, f"GW{gw}", "fixtures.csv")

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

def attach_last_season_features(players_df, base_path="data/2025-26/players"):
    """
    Attach last season stats to players_df by reading each player's history.csv.
    Columns are renamed to match the model's expected last season features.
    Minute buckets are added using bins=[-1, 900, 1800, 2700, 4000].
    Missing files or stats are filled with 0.
    """
    # Mapping from history.csv columns to model feature names
    col_map = {
        "starts": "last_season_starts",
        "minutes": "last_season_minutes",
        "expected_assists": "last_season_expected_assists_per90",
        "expected_goals": "last_season_expected_goals_per90",
        "expected_goal_involvements": "last_season_expected_goal_involvements_per90",
        "expected_goals_conceded": "last_season_expected_goals_conceded_per90",
        "clean_sheets": "last_season_clean_sheets_per90",
        "saves": "last_season_saves_per90",
        "red_cards": "last_season_red_cards_per90",
        "total_points": "last_season_total_points_per90",
        "ict_index": "last_season_ict_index_per90",
        "influence": "last_season_influence_per90",
        "end_cost": "last_season_end_cost"
    }

    bins = [-1, 900, 1800, 2700, 4000]
    labels = ["Low", "Medium", "High", "Ironman"]

    all_features = []

    for idx, row in players_df.iterrows():
        first = row["first_name"].capitalize()
        second = row["second_name"].capitalize()

        # Use glob to find folder even if ID is unknown
        player_folder = os.path.join(base_path, f"{first}_{second}_*")
        folders = glob.glob(player_folder)

        if not folders:
            features = {v: 0 for v in col_map.values()}
            total_minutes = 0
        else:
            folder_path = folders[0]
            history_csv = os.path.join(folder_path, "history.csv")
            if os.path.exists(history_csv):
                history = pd.read_csv(history_csv)
                last_row = history.iloc[-1]  # Last season stats
                features = {col_map[k]: last_row[k] if k in last_row else 0 for k in col_map}
                total_minutes = last_row.get("minutes", 0)
            else:
                features = {v: 0 for v in col_map.values()}
                total_minutes = 0

        # Add minute buckets using your bins
        bucket = pd.cut([total_minutes], bins=bins, labels=labels)[0]
        features["last_season_minutes_bucket_Low"] = int(bucket == "Low")
        features["last_season_minutes_bucket_Medium"] = int(bucket == "Medium")
        features["last_season_minutes_bucket_High"] = int(bucket == "High")
        features["last_season_minutes_bucket_Ironman"] = int(bucket == "Ironman")

        # Preserve metadata
        features["first_name"] = row["first_name"]
        features["second_name"] = row["second_name"]
        features["player_id"] = row["player_id"]
        features["position"] = row["position"]

        all_features.append(features)

    features_df = pd.DataFrame(all_features)

    # Merge with original players_df to preserve order and metadata
    merged_df = players_df.merge(
        features_df,
        on=["player_id", "first_name", "second_name", "position"],
        how="left"
    )

    return merged_df

# Load players
players_raw = pd.read_csv("2025-2026/players.csv")[["player_id", "second_name", "position", "team_code"]]
players = pd.read_csv("2025-2026/players.csv")
players = attach_last_season_features(players)
players = attach_ema_features(players)
players = attach_team_strength(players)

# Position → model suffix mapping
pos_map = {
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD"
}

# Load teams for team names
teams = pd.read_csv("2025-2026/teams.csv")[["code", "name"]]

all_gw_preds = []

for gw in range(2, 12):
    print(f"Predicting GW{gw}...")
    # Attach GW-specific fixture info
    players_gw = attach_gw_fixture_info(players.copy(), gw=gw)
    
    gw_pred_list = []

    for pos_full, pos_suffix in pos_map.items():
        model_path = f"models/hist+form1.json_{pos_suffix}"
        if not os.path.exists(model_path):
            continue

        model = joblib.load(model_path)
        subset = players_gw[players_gw["position"] == pos_full].copy()
        if subset.empty:
            continue

        # Keep metadata
        meta = subset[["player_id", "position", "team_code"]].copy()
        
        # Attach second_name from original players
        meta = meta.merge(players_raw[["player_id", "second_name"]], on="player_id", how="left")
        
        # Merge team name
        meta = meta.merge(teams, left_on="team_code", right_on="code", how="left").rename(columns={"name": "team_name"})
        
        # Only pass feature columns to model
        feature_cols = model.feature_names_in_
        model_input = subset[feature_cols]

        # Predict points
        meta[f"GW{gw}_predicted_points"] = model.predict(model_input)
        gw_pred_list.append(meta)

    # Combine this GW's predictions
    gw_predictions_df = pd.concat(gw_pred_list, ignore_index=True)
    all_gw_preds.append(gw_predictions_df[["player_id", f"GW{gw}_predicted_points"]])

# Merge all GWs together
predictions_df = reduce(lambda left, right: pd.merge(left, right, on="player_id"), all_gw_preds)

# Attach names and team for printing
predictions_df = predictions_df.merge(players_raw, on="player_id", how="left")
predictions_df = predictions_df.merge(teams, left_on="team_code", right_on="code", how="left").rename(columns={"name": "team_name"})

# Compute average points per game
gw_cols = [f"GW{i}_predicted_points" for i in range(2, 12)]
predictions_df["avg_points_per_game"] = predictions_df[gw_cols].mean(axis=1)

# Show only the columns we care about
predictions_df = predictions_df[["second_name", "position", "team_name"] + gw_cols + ["avg_points_per_game"]]

# Sort by average predicted points
predictions_df = predictions_df.sort_values(by="avg_points_per_game", ascending=False)

# positions filter
predictions_df = predictions_df[predictions_df['position'] == 'Goalkeeper']

# Show results
#pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Show all columns
#pd.set_option('display.max_columns', None)

# Show all rows
pd.set_option('display.max_rows', None)

# Prevent truncating column content
pd.set_option('display.max_colwidth', None)

# Expand the display width so columns aren't wrapped
pd.set_option('display.width', 2000)

# Optional: avoid scientific notation for floats
pd.set_option('display.float_format', '{:.2f}'.format)

print(predictions_df.head(50))