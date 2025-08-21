import pandas as pd
import glob
import os
from model_config import NUM_GWS_TO_ROLL

def add_features(data):
    # Grab player ids
    data = data.rename(columns={'element': 'id'})
    
    # Historic Player Performance
    data = add_historic_data(data)

    # 3-GW Rolling Form
    data = add_form_data(data, NUM_GWS_TO_ROLL)

    # Add minutes bucketing (one-hot encoding)
    data = add_minutes_buckets(data)

    # Add fixture information
    data = add_fixture_information(data)

    # Add team form data
    data = add_team_form_data(data)

    # Add player importance ratios
    data = add_player_importance_data(data)
    
    return data

def add_historic_data(data):
    last_season_cache = {}
    stats_cols = [
        'assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
        'expected_assists', 'expected_goal_involvements', 'expected_goals',
        'expected_goals_conceded', 
        'goals_conceded', 'goals_scored',
        'ict_index', 'influence', 'minutes', 'own_goals', 'penalties_missed',
        'penalties_saved', 'red_cards', 'starts', 'threat',
        'total_points', 'yellow_cards', 'end_cost', 'saves'
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
            current_season = data.loc[data['id'] == player_id, 'season_str'].iloc[0]  # get player's season
            player_folder = f"data/{current_season}/players/*_{player_id}"
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

def add_form_data(data, num_gws_to_roll):
    exclude = {"id", "round", "kickoff_time", "modified", "fixture"}
    all_features = [
        "assists","bonus","bps","clean_sheets","creativity",
        "expected_assists", "expected_goal_involvements","expected_goals","expected_goals_conceded",
        "goals_conceded","goals_scored","ict_index","influence","minutes",
        "own_goals","penalties_missed","penalties_saved","red_cards","saves",
        "selected","team_a_score","team_h_score","threat","total_points",
        "transfers_balance","transfers_in","transfers_out","value","yellow_cards", "starts"
    ]
    metrics = [c for c in all_features if c not in exclude]
    form_features = []
    for player_id in data["id"].unique():
        player_data = data.loc[data["id"] == player_id].copy()
        for season in player_data["season_str"].unique():
            player_season_data = player_data[player_data["season_str"] == season]
            player_folder_pattern = f"data/{season}/players/*_{player_id}"
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
                .shift(1)  # don't leak current GW
            )
            rolling = rolling.add_prefix(f"form{num_gws_to_roll}_")
            rolling["round"] = gw_data["round"]
            rolling["id"] = gw_data["id"]
            rolling["season_str"] = season  # keep season info
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
            ema["season_str"] = season  # keep season info
            # Combine
            combined = pd.concat([rolling, ema.drop(columns=["round","id","season_str"])], axis=1)
            form_features.append(combined.reset_index(drop=True))
    if not form_features:  # safeguard if no data found
        return data
    form_features = pd.concat(form_features, ignore_index=True)
    # Merge including season_str to prevent cross-season contamination
    data = data.merge(form_features, on=["id", "round", "season_str"], how="left")
    # Fill missing values
    form_cols = [c for c in data.columns if c.startswith(f"form{num_gws_to_roll}_") or c.startswith(f"ema{num_gws_to_roll}_")]
    data[form_cols] = data[form_cols].fillna(0)
    return data

def add_minutes_buckets(data):
    # Bucket last season minutes into workload tiers
    data["last_season_minutes_bucket"] = pd.cut(
        data["last_season_minutes"],
        bins=[-1, 900, 1800, 2700, 4000],  # <10 full games, ~10–20, ~20–30, 30+
        labels=["Low", "Medium", "High", "Ironman"]
    )
    # Bucket recent EMA minutes (rotation risk proxy)
    data["ema_minutes_bucket"] = pd.cut(
        data[f"ema{NUM_GWS_TO_ROLL}_minutes"],
        bins=[-1, 30, 60, 90, 120],  # bench / sub / starter / always 90+
        labels=["0-30", "30-60", "60-90", "90+"]
    )
    # One-hot encode
    data = pd.get_dummies(
        data, 
        columns=["last_season_minutes_bucket", "ema_minutes_bucket"], 
        drop_first=False
    )
    return data

def add_fixture_information(data):
    data['is_home'] = data['was_home'].astype(int)
    # We'll process each season separately
    all_dfs = []
    for season in data['season_str'].unique():
        season_data = data[data['season_str'] == season].copy()
        team_file = f"data/{season}/teams.csv"
        team_strengths = pd.read_csv(team_file)
        # Merge team strength
        season_data = season_data.merge(
            team_strengths[['name', 'strength']].rename(columns={'name': 'team', 'strength': 'team_strength'}),
            on='team',
            how='left'
        )
        # Merge opponent strength
        season_data = season_data.merge(
            team_strengths[['id', 'strength']].rename(columns={'id': 'opponent_team', 'strength': 'opponent_strength'}),
            on='opponent_team',
            how='left'
        )
        season_data['strength_diff'] = season_data['team_strength'] - season_data['opponent_strength']
        all_dfs.append(season_data)
    return pd.concat(all_dfs, ignore_index=True)

def add_team_form_data(data):
    all_dfs = []
    for season in data['season_str'].unique():
        season_data = data[data['season_str'] == season].copy()
        # Load fixtures + teams for this season
        fixtures_file = f"data/{season}/fixtures.csv"
        teams_file = f"data/{season}/teams.csv"
        fixtures = pd.read_csv(fixtures_file)
        teams = pd.read_csv(teams_file)
        # Build mapping {id -> name}
        team_map = dict(zip(teams['id'], teams['name']))
        # Prepare long format with goals for/against
        home_fixtures = fixtures[['event', 'team_h', 'team_h_score', 'team_a_score']].rename(
            columns={'team_h': 'team', 'team_h_score': 'goals_for', 'team_a_score': 'goals_against', 'event': 'round'}
        )
        away_fixtures = fixtures[['event', 'team_a', 'team_a_score', 'team_h_score']].rename(
            columns={'team_a': 'team', 'team_a_score': 'goals_for', 'team_h_score': 'goals_against', 'event': 'round'}
        )
        all_fixtures = pd.concat([home_fixtures, away_fixtures], ignore_index=True)
        # Map team IDs -> team names to match season_data
        all_fixtures['team'] = all_fixtures['team'].map(team_map)
        all_fixtures = all_fixtures.sort_values(['team', 'round'])
        # Compute EMA using strictly previous gameweeks
        all_fixtures[f'ema{NUM_GWS_TO_ROLL}_team_goals_scored'] = (
            all_fixtures.groupby('team')['goals_for']
            .transform(lambda x: x.shift(1).ewm(span=NUM_GWS_TO_ROLL, adjust=False).mean())
            .fillna(0)
        )
        all_fixtures[f'ema{NUM_GWS_TO_ROLL}_team_goals_conceded'] = (
            all_fixtures.groupby('team')['goals_against']
            .transform(lambda x: x.shift(1).ewm(span=NUM_GWS_TO_ROLL, adjust=False).mean())
            .fillna(0)
        )
        # Merge EMA features into player-level data (team names now match)
        season_data = season_data.merge(
            all_fixtures[['team', 'round',
                          f'ema{NUM_GWS_TO_ROLL}_team_goals_scored',
                          f'ema{NUM_GWS_TO_ROLL}_team_goals_conceded']],
            on=['team', 'round'],
            how='left'
        )
        all_dfs.append(season_data)
    return pd.concat(all_dfs, ignore_index=True)

def add_player_importance_data(data):
    data['player_goals_scored_portion'] = data[f'ema{NUM_GWS_TO_ROLL}_expected_goals'] / data[f'ema{NUM_GWS_TO_ROLL}_team_goals_scored']
    data['player_goals_conceded_portion'] = data[f'ema{NUM_GWS_TO_ROLL}_expected_goals_conceded'] / data[f'ema{NUM_GWS_TO_ROLL}_team_goals_conceded']
    return data