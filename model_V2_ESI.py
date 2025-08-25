import pandas as pd
from features import get_data, POSITION_FEATURE, EARLY_SEASON_INTELLIGENCE, STAT_FEATURES, FPL_FEATURES, TEAM_FEATURES, FIXTURE_FEATURES, LABEL
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import joblib


# MODEL CONFIG
MODEL_NAME = "V2_ESI"

MODEL_V2_ESI_FEATURES = POSITION_FEATURE + EARLY_SEASON_INTELLIGENCE + STAT_FEATURES + FPL_FEATURES + TEAM_FEATURES + FIXTURE_FEATURES

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
    model.fit(train_data[MODEL_V2_ESI_FEATURES], train_data[LABEL])
    return model

# Evaluates the model
def evaluate_model(model, test_data):
    # Make predictions
    predictions = model.predict(test_data[MODEL_V2_ESI_FEATURES])
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

    # Calculate basic regression stats
    mse = mean_squared_error(results["true_points"], results["predicted_points"])
    mae = mean_absolute_error(results["true_points"], results["predicted_points"])
    r2 = r2_score(results["true_points"], results["predicted_points"])
    rho, pval = spearmanr(results["true_points"], results["predicted_points"])
    pd.set_option('display.max_rows', None)
    print("EVALUATION METRICS:")
    print(f"True Average Points: {results["true_points"].mean():.4f}")
    print(f"Predicted Average Points: {results["predicted_points"].mean():.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Spearman Coefficient: {rho:.4f}")

    # Most important features
    importance_dict = model.get_booster().get_score(importance_type='gain')  # gain is usually more informative
    importance_df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    }).sort_values(by='Importance', ascending=False)
    print("\nTop 20 Important Features:")
    print(importance_df.head(20))
    
    # Print a comparison table
    print("\nTRUE VS. PREDICTED:")
    print(results[["player_name", "position", "team_name", "opponent_name", "gw", "status", "form", "minutes_played", "true_points", "predicted_points"]].head(100))

    # Scatter plot
    plt.figure(figsize=(8,6))
    plt.scatter(results["true_points"], results["predicted_points"], alpha=0.5)
    plt.plot([results["true_points"].min(), results["true_points"].max()],
            [results["true_points"].min(), results["true_points"].max()],
            color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('True Points')
    plt.ylabel('Predicted Points')
    plt.title('Predicted vs True FPL Points')
    plt.legend()
    plt.show()


# Saves the model into models/
def save_model(model):
    joblib.dump(model, f"models/{MODEL_NAME}.pkl")

# Trains an XGBoost Regressor model, tests it, and save it to models/
def main():
    # Get data - we generally use gw >= 3 to have recent form data to use
    data = get_data("2024-2025", 1, 38, 0)

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
