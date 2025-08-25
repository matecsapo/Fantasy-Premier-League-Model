import pandas as pd
import glob
import os
import re

# loads in all data into one collective dataframe + converts dates to pandas date objects
def load_data():
    all_dfs = []
    # loop through every season folder
    season_dirs = glob.glob("data/*/gws/")  # e.g. data/2024-25/gws/
    for season_dir in season_dirs:
        # Extract season string from the path
        match_season = re.search(r"data/(\d{4}-\d{2})/gws", season_dir)
        if match_season:
            season_str = match_season.group(1)  # "2024-25"
        else:
            raise ValueError(f"Could not parse season from folder path: {season_dir}")
        # Extract first 2 digits of starting year as int (2024-25 -> 24)
        season_val = int(season_str[:4]) - 2000  # '2024' -> 24
        files = glob.glob(os.path.join(season_dir, "gw*.csv"))
        for f in files:
            # Extract GW number from filename
            match = re.search(r'gw(\d+)\.csv', os.path.basename(f))
            gw_num = int(match.group(1)) if match else None
            # Read CSV ignoring bad lines
            temp_df = pd.read_csv(f, on_bad_lines='skip', encoding='latin1')
            temp_df['GW'] = gw_num
            temp_df['season'] = season_val          # numeric season (24)
            temp_df['season_str'] = season_str      # string season ("2024-25")
            temp_df['kickoff_time'] = pd.to_datetime(temp_df['kickoff_time'], errors='coerce')
            all_dfs.append(temp_df)
    all_dfs = [df.dropna(axis=1, how='all') for df in all_dfs]
    df = pd.concat(all_dfs, ignore_index=True)
    return df
    