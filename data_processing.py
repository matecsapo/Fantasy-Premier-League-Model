import pandas as pd
from config import CURRENT_GAMEWEEK, LOCAL_LOCATION, REMOTE_LOCATION, DATA_SOURCE

# Loads:    player_id | name | team | ...
# order by player_id
def load_players(season):
    location  = ""
    if season == "2024-2025":
        location = f"{DATA_SOURCE}/data/2024-2025/players/players.csv"
    elif season == "2025-2026":
        location = f"{DATA_SOURCE}/data/2025-2026/players.csv"
    players = pd.read_csv(location)
    players = players.sort_values(by="player_id")
    return players

# Loads:    player_id | gw | ... stats ...
# orders by player_id, gw
def load_stats(season, start_gw, end_gw):
    folder = ""
    if season == "2024-2025":
        folder = "playermatchstats"
    elif season == "2025-2026":
        if DATA_SOURCE == LOCAL_LOCATION:
            folder = "By Tournament/Premier League"
        elif DATA_SOURCE == REMOTE_LOCATION:
            folder = "By%20Tournament/Premier%20League"
    gw_stats = []
    for gw in range(start_gw, end_gw + 1):
        this_gw_stats = pd.read_csv(f"{DATA_SOURCE}/data/{season}/{folder}/GW{gw}/playermatchstats.csv")
        this_gw_stats["gw"] = gw
        gw_stats.append(this_gw_stats)
    stats = pd.concat(gw_stats, ignore_index=True)
    stats = stats.sort_values(by=["player_id", "gw"]).reset_index(drop=True)
    return stats

# Loads:    player_id | gw | ... FPL_Info ...
# orders by player_id, gw
def load_FPL_info(season, start_gw, end_gw):
    location = ""
    if season == "2024-2025":
        location = f"{DATA_SOURCE}/data/2024-2025/playerstats/playerstats.csv"
    elif season == "2025-2026":
        location = f"{DATA_SOURCE}/data/2025-2026/playerstats.csv"
    FPL_info = pd.read_csv(location)
    FPL_info = FPL_info[FPL_info["gw"].isin(range(start_gw, end_gw + 1))]
    FPL_info = FPL_info.sort_values(by=["id", "gw"]).reset_index(drop=True)
    return FPL_info

# Loads:    code | ... team_features ...
def load_teams_info(season):
    location = ""
    if season == "2024-2025":
        location = f"{DATA_SOURCE}/data/2024-2025/teams/teams.csv"
    elif season == "2025-2026":
        location = f"{DATA_SOURCE}/data/2025-2026/teams.csv"
    teams_info = pd.read_csv(location)
    return teams_info

# Loads: gw | home_team | away_team | ... match info/stats ...
# orderes by gameweek
def load_fixture_info(season, start_gw, end_gw):
    if season == "2024-2025":
        matches = pd.read_csv(f"{DATA_SOURCE}/data/{season}/matches/matches.csv")
        matches = matches[matches["gameweek"].isin(range(start_gw, end_gw + 1))]
        matches = matches.sort_values(by="gameweek")
        return matches
    elif season == "2025-2026":
        gw_matches = []
        for gw in range(start_gw, end_gw + 1):
            folder = ""
            if DATA_SOURCE == LOCAL_LOCATION:
                folder = "By Tournament/Premier League"
            elif DATA_SOURCE == REMOTE_LOCATION:
                folder = "By%20Tournament/Premier%20League"
            this_gw_matches = pd.read_csv(f"{DATA_SOURCE}/data/2025-2026/{folder}/GW{gw}/matches.csv")
            gw_matches.append(this_gw_matches)
        matches = pd.concat(gw_matches, ignore_index=True)
        matches = matches.sort_values(by="gameweek")
        return matches
