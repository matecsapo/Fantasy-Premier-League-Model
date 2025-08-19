import requests
import os
import pandas as pd
import io

#24/25 data
data_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/"
save_location = "data/24-25"
os.makedirs(save_location, exist_ok=True)
for gw in range(1, 39):
    file_name = f"gw{gw}.csv"
    url = data_URL + file_name
    response = requests.get(url)
    if response.status_code == 200:
        temp_df = pd.read_csv(io.StringIO(response.text))  # or io.StringIO in modern pandas
        # add GW column
        temp_df['GW'] = gw
        # save modified CSV
        temp_df.to_csv(os.path.join(save_location, file_name), index=False)
        print(f"Downloaded {file_name}")
    else:
        print(f"Failed to download {file_name}")
