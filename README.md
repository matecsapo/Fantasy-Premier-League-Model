# Fantasy Premier League Model

This is an ML model for predicting points and picking players in Fantasy Premier League. The model aims to predict points achieved by players and thereby predict picks, captains, transfer, etc...

**Note - All underlying training/gameweek-by-gameweek data is sourced from [olbauday/FPL-Elo-Insights](https://github.com/olbauday/FPL-Elo-Insights).
Massive shout out and thank you for their great work!**

## Streamlit Webapp
  - A minimal app to query model is live at: https://eagleeyefpl.streamlit.app

## Versions
  - V3 - Current/Newest + overall strongest model; trains on previous + current season data
  - V2 - Strongest model trained only on previous season data
  - V2.5 - slight variation on V2; quit hit-or-miss
  - V2_ESI - Adaption of base V2 trained to be perform stronger in early gameweeks when data is scarce + less reliable
  - V1 - Considerably weaker initial test model

## Functionalities:
  - Points predictions for all players for upcoming gameweeks
  - Picking optimal starting 11 from upcoming gameweek onwards
  - (soon) given current team, suggestions for:
      - captaincy
      - transfers
      - substitutions
   
## Predictions:
  - Predictions for upcoming gameweek are automatically refreshed @ 00:00, 06:00, 12:00, 18:00 EST
  - Predictions are organzied by gameweek and model used in the folder predictions/

## Status
Very much so still in early stages - lots of exciting work ahead!
