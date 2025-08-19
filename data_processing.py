import pandas as pd
import glob
import os
import re

# loads in all data into one collective dataframe + converts dates to pandas date objects
def load_data():
    files = glob.glob("data/2024-25/gws/gw*.csv")
    all_dfs = []
    for f in files:
        # Extract GW number from filename
        match = re.search(r'gw(\d+)\.csv', os.path.basename(f))
        gw_num = int(match.group(1)) if match else None
        # Read CSV ignoring bad lines
        temp_df = pd.read_csv(f, on_bad_lines='skip')  # <-- this avoids parsing errors
        temp_df['GW'] = gw_num
        all_dfs.append(temp_df)
    df = pd.concat(all_dfs, ignore_index=True)
    return df
    