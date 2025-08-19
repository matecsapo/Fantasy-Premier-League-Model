import pandas as pd
import glob
import os
from points_model_config import NUM_GWS_TO_ROLL

def add_features(data):
    # Grab player ids
    data = data.rename(columns={'element': 'id'})
    
    # Historic Player Performance
    data = add_historic_player_performance(data)

    # 3-GW Rolling Form
    data = add_form_performance(data, NUM_GWS_TO_ROLL)

    # Add fixture information
    data = add_fixture_information(data)
    
    return data


def add_historic_player_performance(data):
    last_season_cache = {}
    stats_cols = [
        'assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
        'expected_assists', 'expected_goal_involvements', 'expected_goals',
        'expected_goals_conceded', 'goals_conceded', 'goals_scored',
        'ict_index', 'influence', 'minutes', 'own_goals', 'penalties_missed',
        'penalties_saved', 'red_cards', 'saves', 'starts', 'threat',
        'total_points', 'yellow_cards'
    ]
    # Initialize columns
    for col in stats_cols:
        data[f'last_season_{col}'] = 0.0
        if col != 'minutes':  # per-90 metrics don’t include minutes
            data[f'last_season_{col}_per90'] = 0.0
    for player_id in data['id'].unique():
        if player_id in last_season_cache:
            last_stats = last_season_cache[player_id]
        else:
            player_folder = f"data/2024-25/players/*_{player_id}"
            history_files = glob.glob(os.path.join(player_folder, "history.csv"))
            if history_files:
                stats = pd.read_csv(history_files[0])
                last_stats = stats.iloc[-1]  # most recent season
                last_season_cache[player_id] = last_stats
            else:
                last_stats = None
        if last_stats is not None:
            minutes = last_stats.get('minutes', 0)
            # Avoid division by zero
            if pd.isna(minutes) or minutes == 0:
                minutes = 1
            for col in stats_cols:
                value = last_stats.get(col, 0.0)
                data.loc[data['id'] == player_id, f'last_season_{col}'] = value
                if col != 'minutes':
                    data.loc[data['id'] == player_id, f'last_season_{col}_per90'] = value / minutes * 90
    return data

def add_form_performance(data, num_gws_to_roll):
    exclude = {"id", "round", "kickoff_time", "modified", "fixture"}
    all_features = [
        "assists","bonus","bps","clean_sheets","creativity","expected_assists",
        "expected_goal_involvements","expected_goals","expected_goals_conceded",
        "goals_conceded","goals_scored","ict_index","influence","minutes",
        "mng_clean_sheets","mng_draw","mng_goals_scored","mng_loss",
        "mng_underdog_draw","mng_underdog_win","mng_win",
        "own_goals","penalties_missed","penalties_saved","red_cards","saves",
        "selected","starts","team_a_score","team_h_score","threat","total_points",
        "transfers_balance","transfers_in","transfers_out","value","yellow_cards"
    ]
    metrics = [c for c in all_features if c not in exclude]
    form_features = []
    for player_id in data["id"].unique():
        player_folder_pattern = f"data/2024-25/players/*_{player_id}"
        gw_files = glob.glob(os.path.join(player_folder_pattern, "gw.csv"))
        if not gw_files:
            continue
        gw_data = pd.read_csv(gw_files[0]).sort_values("round")
        if "element" in gw_data.columns:
            gw_data = gw_data.rename(columns={"element": "id"})
        # --- Rolling average ---
        rolling = (
            gw_data.set_index("round")[metrics]
            .rolling(window=num_gws_to_roll, min_periods=1)
            .mean()
        )
        rolling = rolling.shift(1)  # don't leak current GW
        rolling = rolling.add_prefix(f"form{num_gws_to_roll}_")
        rolling["round"] = gw_data["round"]
        rolling["id"] = gw_data["id"]
        # --- Exponential moving average ---
        ema = (
            gw_data.set_index("round")[metrics]
            .ewm(span=num_gws_to_roll, adjust=False)
            .mean()
            .shift(1)
        )
        ema = ema.add_prefix(f"ema{num_gws_to_roll}_")
        ema["round"] = gw_data["round"]
        ema["id"] = gw_data["id"]
        # Combine
        combined = pd.concat([rolling, ema.drop(columns=["round","id"])], axis=1)
        form_features.append(combined.reset_index(drop=True))
    if not form_features:  # safeguard if no data found
        return data
    form_features = pd.concat(form_features, ignore_index=True)
    data = data.merge(form_features, on=["id", "round"], how="left")
    # Fill missing values
    form_cols = [c for c in data.columns if c.startswith(f"form{num_gws_to_roll}_") or c.startswith(f"ema{num_gws_to_roll}_")]
    data[form_cols] = data[form_cols].fillna(0)
    return data

def add_fixture_information(data):
    data['is_home']= data['was_home'].astype(int)

    team_strengths = pd.read_csv("data/2024-25/teams.csv")
    data = data.merge(
        team_strengths[['name', 'strength']].rename(columns={'name': 'team', 'strength': 'team_strength'}),
        on='team',
        how='left'
    )
    data = data.merge(
        team_strengths[['id', 'strength']].rename(columns={'id': 'opponent_team', 'strength': 'opponent_strength'}),
        on='opponent_team',
        how='left'
    )
    return data