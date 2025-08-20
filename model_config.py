MODEL_NAME = "test.json"

MODEL_PATH = f'models/{MODEL_NAME}'

NUM_GWS_TO_ROLL = 3

# Random Forest Specifications
NUM_ESTIMATORS = 500

MAX_DEPTH = 10

MIN_SAMPLES_SPLIT = 5

FEATURES = [
    # Playtime ()
    #'last_season_starts',
    'last_season_minutes',
    'last_season_minutes_bucket_Low',
    'last_season_minutes_bucket_Medium',
    'last_season_minutes_bucket_High',
    'last_season_minutes_bucket_Ironman',
    f'ema{NUM_GWS_TO_ROLL}_minutes',
    'ema_minutes_bucket_0-30',
    'ema_minutes_bucket_30-60',
    'ema_minutes_bucket_60-90',
    'ema_minutes_bucket_90+',

    # PLAYER UNDERSTATS
    # Attacking metrics
    #'last_season_assists_per90',
    'last_season_expected_assists_per90',
    #'last_season_goals_scored_per90',
    'last_season_expected_goals_per90',
    'last_season_expected_goal_involvements_per90',
    #'last_season_penalties_missed_per90',
    #f'ema{NUM_GWS_TO_ROLL}_assists',
    f'ema{NUM_GWS_TO_ROLL}_expected_assists',
    #f'ema{NUM_GWS_TO_ROLL}_goals_scored',
    f'ema{NUM_GWS_TO_ROLL}_expected_goals',
    f'ema{NUM_GWS_TO_ROLL}_expected_goal_involvements',
    #f'ema{NUM_GWS_TO_ROLL}_penalties_missed',
    # Defensive Metrics
    #'last_season_goals_conceded_per90',
    'last_season_expected_goals_conceded_per90',
    #'last_season_clean_sheets_per90',
    #'last_season_saves_per90',
    #'last_season_penalties_saved_per90',
    #'last_season_own_goals_per90',
    #'last_season_red_cards_per90',
    #'last_season_yellow_cards_per90',
    #f'ema{NUM_GWS_TO_ROLL}_goals_conceded',
    f'ema{NUM_GWS_TO_ROLL}_expected_goals_conceded',
    #f'ema{NUM_GWS_TO_ROLL}_clean_sheets',
    #f'ema{NUM_GWS_TO_ROLL}_saves',
    #f'ema{NUM_GWS_TO_ROLL}_penalties_saved',
    #f'ema{NUM_GWS_TO_ROLL}_own_goals',
    #f'ema{NUM_GWS_TO_ROLL}_red_cards',
    #f'ema{NUM_GWS_TO_ROLL}_yellow_cards',

    # Points History
    'last_season_total_points_per90',
    #'last_season_ict_index_per90',
    #'last_season_influence_per90',
    'last_season_end_cost',
    f'ema{NUM_GWS_TO_ROLL}_total_points',
    #f'ema{NUM_GWS_TO_ROLL}_ict_index',
    #f'ema{NUM_GWS_TO_ROLL}_influence',
    f'ema{NUM_GWS_TO_ROLL}_value',

    # Team Understats
    'team_strength',

    # Fixture Difficulty
    'is_home',
    'opponent_strength'
]

OLD_FEATURES = [
    # Expected Values (Via Opta)
    #'xP',
    #'expected_assists',
    #'expected_goals',
    #'expected_goal_involvements',
    #'expected_goals_conceded',

    # Historic (Last Season) Player Performance
    # FPL Point Metrics
    'last_season_total_points',
    'last_season_total_points_per90',
    #'last_season_bps',
    #'last_season_bps_per90',
    #'last_season_bonus',
    #'last_season_bonus_per90',
    #'last_season_ict_index',
    'last_season_ict_index_per90',
    #'last_season_influence',
    'last_season_influence_per90',

    # Play Frequency
    'last_season_starts',
    'last_season_minutes',

    # Attacking Metrics
    #'last_season_assists',
    'last_season_assists_per90',
    #'last_season_expected_assists',
    'last_season_expected_assists_per90',
    #'last_season_goals_scored',
    'last_season_goals_scored_per90',
    #'last_season_expected_goals',
    'last_season_expected_goals_per90',
    #'last_season_expected_goal_involvements',
    'last_season_expected_goal_involvements_per90',
    #'last_season_penalties_missed',
    'last_season_penalties_missed_per90',

    # Defensive Metrics
    #'last_season_goals_conceded',
    'last_season_goals_conceded_per90',
    #'last_season_expected_goals_conceded',
    'last_season_expected_goals_conceded_per90',
    #'last_season_clean_sheets',
    'last_season_clean_sheets_per90',
    #'last_season_saves',
    'last_season_saves_per90',
    #'last_season_penalties_saved',
    'last_season_penalties_saved_per90',
    #'last_season_own_goals',
    'last_season_own_goals_per90',
    #'last_season_red_cards',
    'last_season_red_cards_per90',
    #'last_season_yellow_cards',
    'last_season_yellow_cards_per90',

    # Current Player Form
    # Simple Rolling
    #f'form{NUM_GWS_TO_ROLL}_assists',
    #f'form{NUM_GWS_TO_ROLL}_bonus',
    #f'form{NUM_GWS_TO_ROLL}_bps',
    #f'form{NUM_GWS_TO_ROLL}_clean_sheets',
    #f'form{NUM_GWS_TO_ROLL}_creativity',
    #f'form{NUM_GWS_TO_ROLL}_expected_assists',
    #f'form{NUM_GWS_TO_ROLL}_expected_goal_involvements',
    #f'form{NUM_GWS_TO_ROLL}_expected_goals',
    #f'form{NUM_GWS_TO_ROLL}_expected_goals_conceded',
    #f'form{NUM_GWS_TO_ROLL}_goals_conceded',
    #f'form{NUM_GWS_TO_ROLL}_goals_scored',
    #f'form{NUM_GWS_TO_ROLL}_ict_index',
    #f'form{NUM_GWS_TO_ROLL}_influence',
    #f'form{NUM_GWS_TO_ROLL}_minutes',
    #f'form{NUM_GWS_TO_ROLL}_mng_clean_sheets',
    #f'form{NUM_GWS_TO_ROLL}_mng_draw',
    #f'form{NUM_GWS_TO_ROLL}_mng_goals_scored',
    #f'form{NUM_GWS_TO_ROLL}_mng_loss',
    #f'form{NUM_GWS_TO_ROLL}_mng_underdog_draw',
    #f'form{NUM_GWS_TO_ROLL}_mng_underdog_win',
    #f'form{NUM_GWS_TO_ROLL}_mng_win',
    #f'form{NUM_GWS_TO_ROLL}_own_goals',
    #f'form{NUM_GWS_TO_ROLL}_penalties_missed',
    #f'form{NUM_GWS_TO_ROLL}_penalties_saved',
    #f'form{NUM_GWS_TO_ROLL}_red_cards',
    #f'form{NUM_GWS_TO_ROLL}_saves',
    #f'form{NUM_GWS_TO_ROLL}_selected',
    #f'form{NUM_GWS_TO_ROLL}_starts',
    #f'form{NUM_GWS_TO_ROLL}_team_a_score',
    #f'form{NUM_GWS_TO_ROLL}_team_h_score',
    #f'form{NUM_GWS_TO_ROLL}_threat',
    #f'form{NUM_GWS_TO_ROLL}_total_points',
    #f'form{NUM_GWS_TO_ROLL}_transfers_balance',
    #f'form{NUM_GWS_TO_ROLL}_transfers_in',
    #f'form{NUM_GWS_TO_ROLL}_transfers_out',
    #f'form{NUM_GWS_TO_ROLL}_value',
    #f'form{NUM_GWS_TO_ROLL}_yellow_cards',
    # EMA Rolling
    f'ema{NUM_GWS_TO_ROLL}_assists',
    #f'ema{NUM_GWS_TO_ROLL}_bonus',
    #f'ema{NUM_GWS_TO_ROLL}_bps',
    f'ema{NUM_GWS_TO_ROLL}_clean_sheets',
    f'ema{NUM_GWS_TO_ROLL}_creativity',
    f'ema{NUM_GWS_TO_ROLL}_expected_assists',
    f'ema{NUM_GWS_TO_ROLL}_expected_goal_involvements',
    f'ema{NUM_GWS_TO_ROLL}_expected_goals',
    f'ema{NUM_GWS_TO_ROLL}_expected_goals_conceded',
    f'ema{NUM_GWS_TO_ROLL}_goals_conceded',
    f'ema{NUM_GWS_TO_ROLL}_goals_scored',
    f'ema{NUM_GWS_TO_ROLL}_ict_index',
    f'ema{NUM_GWS_TO_ROLL}_influence',
    f'ema{NUM_GWS_TO_ROLL}_minutes',
    #f'ema{NUM_GWS_TO_ROLL}_mng_clean_sheets',
    #f'ema{NUM_GWS_TO_ROLL}_mng_draw',
    #f'ema{NUM_GWS_TO_ROLL}_mng_goals_scored',
    #f'ema{NUM_GWS_TO_ROLL}_mng_loss',
    #f'ema{NUM_GWS_TO_ROLL}_mng_underdog_draw',
    #f'ema{NUM_GWS_TO_ROLL}_mng_underdog_win',
    #f'ema{NUM_GWS_TO_ROLL}_mng_win',
    f'ema{NUM_GWS_TO_ROLL}_own_goals',
    f'ema{NUM_GWS_TO_ROLL}_penalties_missed',
    f'ema{NUM_GWS_TO_ROLL}_penalties_saved',
    f'ema{NUM_GWS_TO_ROLL}_red_cards',
    f'ema{NUM_GWS_TO_ROLL}_saves',
    f'ema{NUM_GWS_TO_ROLL}_selected',
    f'ema{NUM_GWS_TO_ROLL}_starts',
    #f'ema{NUM_GWS_TO_ROLL}_team_a_score',
    #f'ema{NUM_GWS_TO_ROLL}_team_h_score',
    f'ema{NUM_GWS_TO_ROLL}_threat',
    f'ema{NUM_GWS_TO_ROLL}_total_points',
    #f'ema{NUM_GWS_TO_ROLL}_transfers_balance',
    #f'ema{NUM_GWS_TO_ROLL}_transfers_in',
    #f'ema{NUM_GWS_TO_ROLL}_transfers_out',
    f'ema{NUM_GWS_TO_ROLL}_value',
    f'ema{NUM_GWS_TO_ROLL}_yellow_cards',

    # Historic Team Performance

    # Current Team Form

    # Fixture Difficulty
    'is_home',
    'team_strength',
    'opponent_strength'
]

# all features organically in data set
# name,position,team,xP,assists,bonus,bps,clean_sheets,creativity,element,expected_assists,expected_goal_involvements,expected_goals,expected_goals_conceded,fixture,goals_conceded,goals_scored,ict_index,influence,kickoff_time,minutes,modified,opponent_team,own_goals,penalties_missed,penalties_saved,red_cards,round,saves,selected,starts,team_a_score,team_h_score,threat,total_points,transfers_balance,transfers_in,transfers_out,value,was_home,yellow_cards
# Alex Scott,MID,Bournemouth,1.6,0,0,11,0,12.8,77,0.01,0.01,0.00,1.02,6,1,0,3.6,22.8,2024-08-17T14:00:00Z,62,False,16,0,0,0,0,1,0,4339,1,1,1,0.0,2,0,0,0,50,False,0