# Imports config.json settings into python env
import json
with open("config.json") as f:
    config = json.load(f)

# Current gameweek we are predicting for
CURRENT_GAMEWEEK = config["current_gameweek"]

# Data source locations
LOCAL_LOCATION = config["local_location"] # folder name of local fork of FPL-Core-Insights github repo
REMOTE_LOCATION = config["remote_location"] # github access point of FPL-Core-Insights repo

# Selected data source to pull data from
DATA_SOURCE = ""
if config["data_source"] == "local":
    DATA_SOURCE = LOCAL_LOCATION
else:
    DATA_SOURCE = REMOTE_LOCATION