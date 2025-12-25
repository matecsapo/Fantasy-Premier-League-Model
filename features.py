import pandas as pd
from data_processing import load_players, load_stats, load_FPL_info, load_teams_info, load_fixture_info

IDENTIFYING_FEATURES = [
    "player_id",
    "gw",
    "first_name",
    "second_name",
    "position",
    "minutes_played",
    "status",
    "season_minutes_portion",
    "team_code",
    "team_name",
    "opponent_code",
    "opponent_name",
]

POSITION_FEATURE = [
    # One-hot encoding of position
    "position_Goalkeeper", "position_Defender", "position_Midfielder", "position_Forward"
]

EARLY_SEASON_INTELLIGENCE = ["early_season_flag"]

STAT_FEATURES = [
    # Playtime metrics
    # One-hot encoding of portion of minutes of team's season played so far
    "season_minutes_portion_Low", "season_minutes_portion_Medium", "season_minutes_portion_High", "season_minutes_portion_Ironman",

    # Attacking metrics
    "season_goals_per90",
    "season_assists_per90",
    "season_total_shots_per90",
    "season_xg_per90",
    "season_xa_per90",
    "season_xgot_per90",
    "season_shots_on_target_per90",
    "season_chances_created_per90",
    #"season_penalties_scored_per90",

    # Passing and progression metrics
    "season_touches_per90",
    "season_accurate_passes_per90",
    "season_final_third_passes_per90",
    "season_accurate_crosses_per90",
    "season_accurate_long_balls_per90",
    "season_successful_dribbles_per90",

    # Defensive metrics
    "season_tackles_won_per90",
    "season_interceptions_per90",
	"season_recoveries_per90",
	"season_clearances_per90",
	"season_dribbled_past_per90",
	"season_duels_won_per90",
	"season_fouls_committed_per90",

    # Goalkeeping metrics
    "season_saves_per90",
    "season_goals_conceded_per90",
    "season_xgot_faced_per90",
    "season_goals_prevented_per90",

    # Team-wide metrics
    "season_team_goals_conceded_per90"
]

FPL_FEATURES = [
    # Base features (via API)
    "chance_of_playing_this_round",
    "now_cost",
    "selected_by_percent",
    "form",
    "form_rank",
    "value_form",
    "value_season",
    "points_per_game",

    # Engineered Features
    # One-hot encoding of status
    "status_a", "status_d", "status_i", "status_u"
    # points performance
    #"season_points_per90",
    #"form_points"
]

TEAM_FEATURES = [
    "team_strength",
    "team_elo",
]

FIXTURE_FEATURES = [
    "is_home",
    "team_fixture_elo",
    "opponent_fixture_elo",
    "fixture_elo_difference",
    "opponent_strength",
    "opponent_elo"
]

LABEL = "points"

DATA_FEATURES = IDENTIFYING_FEATURES + POSITION_FEATURE + EARLY_SEASON_INTELLIGENCE + STAT_FEATURES + FPL_FEATURES + TEAM_FEATURES + FIXTURE_FEATURES + [LABEL]

# Creates all features for all desired datapoints
# player_id | gw | ... features ...
def get_data(season, start_gw, end_gw, oppenent_game_relative_num, FPL_data_shit):
    # prepare dataset
    players_gws = []
    for gw in range(start_gw, end_gw + 1):
        this_player_gw = load_players(season)
        this_player_gw["gw"] = gw
        players_gws.append(this_player_gw)
    datapoints = pd.concat(players_gws, ignore_index=True)
    
    # Engineer all features
    # Positions feature
    position_dummies = pd.get_dummies(datapoints["position"], prefix="position")
    datapoints = pd.concat([datapoints, position_dummies], axis=1)

    # Early season flag / intelligence feature
    datapoints["early_season_flag"] = (datapoints["gw"] <= 3).astype(int)

    # Stats features
    stats_features = get_stats_features(season)
    datapoints = pd.merge(datapoints, stats_features, on=["player_id", "gw"], how="left")

    # FPL features
    fpl_features = get_FPL_features(season, FPL_data_shit)
    fpl_features.to_csv("test.csv")
    datapoints = pd.merge(datapoints, fpl_features, on=["player_id", "gw"], how="left")

    # Team features
    team_features = get_team_features(season)
    datapoints = pd.merge(datapoints, team_features, on="team_code", how="left")

    # Fixture features
    fixture_features = get_fixture_features(season, start_gw, end_gw, oppenent_game_relative_num)
    datapoints = pd.merge(datapoints, fixture_features, on=["team_code", "gw"], how="left")

    # Keep only the features we need
    datapoints = datapoints[DATA_FEATURES]

    # Remove managers
    datapoints = datapoints[datapoints["position"] != "Unknown"]

    return datapoints

# Returns the FPL-related features associated with player + gw datapoints
# player_id | gw | ... FPL_features ...
def get_FPL_features(season, FPL_data_shift):
    features = load_FPL_info(season, 0, 38)
    features = features.rename(columns={"id": "player_id"})
    features = features.rename(columns={"event_points" : "points"})
    # Make sure we have all player x gw combos
    players_gws = []
    for gw in range(0, 38 + 1):
        this_player_gw = load_players(season)
        this_player_gw["gw"] = gw
        players_gws.append(this_player_gw)
    all_datapoints = pd.concat(players_gws, ignore_index=True)
    features = pd.merge(all_datapoints, features, on=["player_id", "gw"], how="left")
    features = features.sort_values(["player_id", "gw"]).reset_index(drop=True)
    # Calculate season_points_per90 and form_points
    #########
    # Shift gameweek-dependent features if needed to ensure we don't leak future data
    if FPL_data_shift:
        shift_features = ["status", "now_cost", "selected_by_percent", "form", "form_rank", "value_form", "value_season", "points_per_game"]
        features[shift_features] = features.groupby("player_id")[shift_features].transform(lambda x: x.shift(1))
    # Add one-hot-encoding of availability status
    status_dummies = pd.get_dummies(features['status'], prefix='status')
    features = pd.concat([features, status_dummies], axis=1)
    bool_cols = features.select_dtypes(include='bool').columns
    features[bool_cols] = features[bool_cols].astype(int)
    # Keep only the features we need
    features = features[["player_id", "gw", "status"] + FPL_FEATURES + [LABEL]]
    return features

# Returns the team-related features associated with each team
# team_code | team_name | team_strength | team_elo 
def get_team_features(season):
    teams_info = load_teams_info(season)
    # Rename features
    teams_info = teams_info.rename(columns={"code": "team_code"})
    teams_info = teams_info.rename(columns={"name": "team_name"})
    teams_info = teams_info.rename(columns={"strength": "team_strength"})
    teams_info = teams_info.rename(columns={"elo": "team_elo"})
    # Keep only the features we need
    teams_info = teams_info[["team_code", "team_name"] + TEAM_FEATURES]
    return teams_info

def get_fixture_features(season, start_gw, end_gw, opponent_game_relative_num):
    matches = load_fixture_info(season, 1, 38)
    matches = matches.rename(columns={"gameweek": "gw"})
    # Keep only the info we need
    matches = matches[["gw", "home_team", "away_team", "home_team_elo", "away_team_elo"]]
    # Generalize away home / away to find player's team
    home = matches.assign(
        team_code = matches["home_team"],
        team_fixture_elo = matches["home_team_elo"],
        opponent_code = matches["away_team"],
        opponent_fixture_elo = matches["away_team_elo"],
        is_home = 1
    )
    away = matches.assign(
        team_code = matches["away_team"],
        team_fixture_elo = matches["away_team_elo"],
        opponent_code = matches["home_team"],
        opponent_fixture_elo = matches["home_team_elo"],
        is_home = 0
    )
    all_matches = pd.concat([home, away], ignore_index=True)
    all_matches["fixture_elo_difference"] = all_matches["team_fixture_elo"] - all_matches["opponent_fixture_elo"]
    # Grab opponent name and strength
    teams_info = load_teams_info(season)
    teams_info = teams_info.rename(columns={"code": "opponent_code"})
    teams_info = teams_info.rename(columns={"name": "opponent_name"})
    teams_info = teams_info.rename(columns={"strength": "opponent_strength"})
    teams_info = teams_info.rename(columns={"elo": "opponent_elo"})
    teams_info = teams_info[["opponent_code", "opponent_name", "opponent_strength", "opponent_elo"]]
    all_matches = pd.merge(all_matches, teams_info, on="opponent_code", how="left")
    # Shift to get correct gameweek's opponent
    all_matches["gw"] = all_matches["gw"] - opponent_game_relative_num
    return all_matches

def get_stats_features(season):
    stats = load_stats(season, 1, 38)
    # stats only includes gw instances in which player plays >= 1 minute
    # we 0-fill for player that didn't play!
    players_gws = []
    for gw in range(1, 38 + 1):
        this_player_gw = load_players(season)
        this_player_gw["gw"] = gw
        players_gws.append(this_player_gw)
    all_stats = pd.concat(players_gws, ignore_index=True)
    stats = pd.merge(all_stats, stats, on=["player_id", "gw"], how="left")
    stats = stats.fillna(0)
    stats = stats.sort_values(["player_id", "gw"])
    # Season-up-to-now stats
    # Portion of team's minutes played
    #stats["season_minutes"] = stats.groupby("player_id", group_keys=False)["minutes_played"].cumsum().shift(fill_value=0)
    stats["season_minutes"] = stats.groupby("player_id")["minutes_played"].transform(lambda x: x.cumsum().shift(fill_value=0))
    stats["season_minutes_portion"] = stats["season_minutes"] / ((stats["gw"] - 1) * 90)
    stats["season_minutes_bucket"] = pd.cut(
        stats["season_minutes_portion"],
        bins=[-float("inf"), 0.25, 0.65, 0.90, float("inf")],
        labels=["Low", "Medium", "High", "Ironman"]
    )
    minutes_dummies = pd.get_dummies(stats["season_minutes_bucket"], prefix="season_minutes_portion")
    stats = pd.concat([stats, minutes_dummies], axis=1)
    # Normalized-per90 stats
    cumulative_features = [f.replace("_per90", "") for f in STAT_FEATURES[4:]]
    for feature in cumulative_features:
        #stats[f"{feature}"] = stats.groupby("player_id", group_keys=False)[f"{feature.replace("season_", "")}"].cumsum().shift(fill_value=0)
        stats[f"{feature}"] = stats.groupby("player_id")[f"{feature.replace('season_', '')}"].transform(lambda x: x.cumsum().shift(fill_value=0))
    normalized_features = STAT_FEATURES[4:]
    for n_feature in normalized_features:
        stats[f"{n_feature}"] = (stats[f"{n_feature.replace('_per90', '')}"] / stats["season_minutes"]) * 90
    
    # FORM-ROLLING-EMA STATS
    ###########

    # Keep only the columns we need
    stats = stats[["player_id", "gw", "minutes_played", "season_minutes_portion"] + STAT_FEATURES]

    return stats


#data = get_data("2024-2025", 1, 38, 0)
#data = data.sort_values(by=["player_id", "gw"])
#data.to_csv("features.csv")

#data = get_data("2025-2026", 4, 4, 0, False)
#data = data.sort_values(by=["player_id", "gw"])
#data.to_csv("features.csv") 
