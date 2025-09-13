import pandas as pd
from features import get_data, POSITION_FEATURE, STAT_FEATURES, FPL_FEATURES, TEAM_FEATURES, FIXTURE_FEATURES, LABEL
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import joblib
import os


# MODEL CONFIG
MODEL_NAME = "V2_5"

MODEL_V2_5_FEATURES = [
    # Position Features
    "position_Goalkeeper", "position_Defender", "position_Midfielder", "position_Forward",

    # (Under)Stat features
    "season_xg_per90",
    "season_xa_per90",
    "season_team_goals_conceded_per90",
    "season_xgot_faced_per90",
    "season_saves_per90",

    # Team features
    "team_fixture_elo",

    # Fixture features
    "is_home",
    "opponent_strength",
    "opponent_fixture_elo",
    "fixture_elo_difference",

    # FPL features
    "status_a", "status_d", "status_i", "status_u",
    "chance_of_playing_this_round",
    "form",
    "points_per_game",
]

NUM_ESTIMATORS = 200

# Splits data into training and testing data
def split_train_test(data):
    train_data, test_data = train_test_split(
        data, 
        test_size=0.2,    # 20% goes to testing
        random_state=42,  # ensures reproducibility
        shuffle=True      # shuffles data before splitting (default is True)
    )
    return train_data, test_data

# Trains an XGBoost Regressor model
def train_model(train_data):
    model = XGBRegressor(
        n_estimators=NUM_ESTIMATORS,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(train_data[MODEL_V2_5_FEATURES], train_data[LABEL])
    return model

# Evaluates the model
def evaluate_model(model, test_data):
    # Make predictions
    predictions = model.predict(test_data[MODEL_V2_5_FEATURES])
    results = pd.DataFrame({
        "player_name" : test_data["second_name"],
        "position" : test_data["position"],
        "team_name" : test_data["team_name"],
        "opponent_name" : test_data["opponent_name"],
        "gw" : test_data["gw"],
        "status" : test_data["status"],
        "form" : test_data["form"],
        "minutes_played" : test_data["minutes_played"],
        "true_points" : test_data["points"],
        "predicted_points" : predictions
    })
    results = results.sort_values(by="true_points", ascending=False)

    # Folder into which to store the results
    os.makedirs(f"models/{MODEL_NAME}/", exist_ok=True)

    # Calculate basic regression stats
    mse = mean_squared_error(results["true_points"], results["predicted_points"])
    mae = mean_absolute_error(results["true_points"], results["predicted_points"])
    r2 = r2_score(results["true_points"], results["predicted_points"])
    rho, pval = spearmanr(results["true_points"], results["predicted_points"])
    evaluation_metrics = pd.DataFrame({
        "Metric": [
            "True Average Points",
            "Predicted Average Points",
            "Mean Squared Error (MSE)",
            "Mean Absolute Error (MAE)",
            "R^2 Score",
            "Spearman Coefficient"
        ],
        "Value": [
            results['true_points'].mean(),
            results['predicted_points'].mean(),
            mse,
            mae,
            r2,
            rho
        ]
    })
    evaluation_metrics["Value"] = evaluation_metrics["Value"].round(4)
    evaluation_metrics.to_csv(f"models/{MODEL_NAME}/Evaluation_Metrics.csv", index=False)

    # Most important features
    importance_dict = model.get_booster().get_score(importance_type='gain')  # gain is usually more informative
    importance_df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    }).sort_values(by='Importance', ascending=False)
    importance_df.head(20).to_csv(f"models/{MODEL_NAME}/Feature_Importances.csv")

# Saves the model into models/
def save_model(model):
    joblib.dump(model, f"models/{MODEL_NAME}/{MODEL_NAME}.pkl")

# Trains an XGBoost Regressor model, tests it, and save it to models/
def main():
    # Get data - we generally use gw >= 3 to have recent form data to use
    data = get_data("2024-2025", 4, 38, 0, True)

    # Keep only rows with valid points label
    data = data[data[LABEL].notna()]

    # Split into training / testing data
    train_data, test_data = split_train_test(data)

    # Train model
    model = train_model(train_data)

    # Evaluate model
    evaluate_model(model, test_data)

    # Save model
    save_model(model)


if __name__ == "__main__":
    main()
