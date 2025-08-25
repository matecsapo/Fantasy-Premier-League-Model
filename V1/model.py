import pandas as pd
import numpy as np
import glob
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from scipy.stats import spearmanr
import joblib
from model_config import FEATURES, MODEL_PATH, MODEL_NAME, NUM_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT, TRAIN_SEASONS, TEST_SEASONS
import matplotlib.pyplot as plt
import seaborn as sns

#def split_train_test(data):
#    data = data[data['GW'] >= 5]
#    train_data = data[data['season'].isin(TRAIN_SEASONS)]
#    test_data = data[data['season'].isin(TEST_SEASONS)]
#    return train_data, test_data 

def split_train_test(data):
    data = data[data['GW'] >= 5]

    # Only keep players with both files
    def has_files(season, pid):
        folder = f"data/{season}/players/*_{pid}"
        return (
            glob.glob(os.path.join(folder, "history.csv")) and 
            glob.glob(os.path.join(folder, "gw.csv"))
        )

    valid = [pid for season in data['season_str'].unique()
                  for pid in data.loc[data['season_str'] == season, 'id'].unique()
                  if has_files(season, pid)]

    data = data[data['id'].isin(valid)]

    train_data = data[data['season'].isin(TRAIN_SEASONS)]
    test_data = data[data['season'].isin(TEST_SEASONS)]
    return train_data, test_data

#def train_model(train_input, train_output):
#    model = LinearRegression()
#    model.fit(train_input, train_output)
#   return model

def train_model(train_data, model_type = "linear", num_estimators = NUM_ESTIMATORS):
    # different models for each position
    positions = ['GK', 'DEF', 'MID', 'FWD']
    models = {}
    for pos in positions:
        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "randomforest":
            model = RandomForestRegressor(
                n_estimators=num_estimators, 
                random_state=42,
                max_depth = MAX_DEPTH,
                min_samples_split = MIN_SAMPLES_SPLIT
                )
        elif model_type == "xgboostregressor":
            model = XGBRegressor(
                n_estimators=500,      # number of boosting rounds (trees)
                learning_rate=0.05,    # step size shrinkage
                max_depth=5,           # depth of individual trees
                subsample=0.8,         # row sampling (like RF bootstrap)
                colsample_bytree=0.8,  # feature sampling
                reg_lambda=1,          # L2 regularization
                random_state=42,
                n_jobs=-1,
                tree_method="hist",     # faster training on large datasets
            )
        pos_data = train_data[train_data['position'] == pos]
        model.fit(pos_data[FEATURES], pos_data['total_points'])
        models[pos] = model
    return models

def evaluate_model(models, test_data):
    all_model_predictions = []
    all_true_values = []
    all_player_names = []   # <-- keep names too
    # extra: initialize dictionary to collect feature coefficients
    feature_importances = {f: [] for f in FEATURES}
    for pos, model in models.items():
        pos_data = test_data[test_data['position'] == pos]
        predictions = model.predict(pos_data[FEATURES])
        all_model_predictions.extend(predictions)
        all_true_values.extend(pos_data['total_points'])
        all_player_names.extend(pos_data['name'])   # <-- player names
        # extra: collect linear model coefficients if available
        if hasattr(model, 'coef_'):
            for f, coef in zip(FEATURES, model.coef_):
                feature_importances[f].append(abs(coef))
    model_predictions = np.array(all_model_predictions)
    true_values = np.array(all_true_values)
    # model performance
    mean_abs_err = mean_absolute_error(true_values, model_predictions)
    mean_sqr_err = mean_squared_error(true_values, model_predictions)
    root_mean_sqr_err = root_mean_squared_error(true_values, model_predictions)
    r2 = 1 - (np.sum((true_values - model_predictions) ** 2) /
              np.sum((true_values - np.mean(true_values)) ** 2))
    print("=== MODEL PERFORMANCE ===")
    print(f"True Avg Points: {true_values.mean():.2f}")
    print(f"Predicted Avg Points : {model_predictions.mean():.2f}")
    print(f"MAE : {mean_abs_err:.2f}")
    print(f"MSE : {mean_sqr_err:.2f}")
    print(f"RMSE : {root_mean_sqr_err:.2f}")
    print(f"Error Ratio : {100 * mean_abs_err / true_values.mean():.2f}%")
    print(f"R² score : {r2:.3f}")
    # Spearman rank correlation
    spearman_corr, _ = spearmanr(true_values, model_predictions)
    print(f"Spearman Rank Correlation: {spearman_corr:.3f}")
    # extra: feature importance summary
    avg_importances = {f: np.mean(coefs) for f, coefs in feature_importances.items() if coefs}
    sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)
    print("\nTOP 10 MOST IMPORTANT FEATURES (avg. abs. coef across positions):")
    for f, val in sorted_features[:10]:
        print(f"{f}: {val:.4f}")
    # extra: correlation with predictions
    df = test_data[test_data['position'].isin(models.keys())][FEATURES].copy()
    df['pred'] = model_predictions
    corr_with_pred = df.corr()['pred'].drop('pred').abs().sort_values(ascending=False)
    print("\nTOP 10 FEATURES STRONGLY CORRELATED WITH PREDICTIONS:")
    print(corr_with_pred.head(10))
    # first 100 non-zero true vs. predicted sample
    results_df = pd.DataFrame({
        'Player': all_player_names,
        'True Points': true_values,
        'Predicted Points': model_predictions
    })
    # Compute per-player averages
    player_avg = results_df.groupby('Player').agg({
        'True Points': 'mean',
        'Predicted Points': 'mean'
    }).reset_index()
    # Scatter plot of True vs Predicted for all players
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=player_avg['True Points'],
        y=player_avg['Predicted Points']
    )
    plt.plot([player_avg['True Points'].min(), player_avg['True Points'].max()],
            [player_avg['True Points'].min(), player_avg['True Points'].max()],
            color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Average True Points')
    plt.ylabel('Average Predicted Points')
    plt.title('Predicted vs True Points for All Players')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Sort by average true points
    top_avg_players = player_avg.sort_values('True Points', ascending=False).head(100)
    spearman_corr, _ = spearmanr(top_avg_players['True Points'], top_avg_players['Predicted Points'])
    print(f"\nSpearman rank correlation (Top 100 players): {spearman_corr:.3f}")
    print("\nPREDICTED VS. TRUE POINTS (Top 100 Players by Average True Points)")
    with pd.option_context('display.max_rows', 100, 'display.max_columns', None):
        print(top_avg_players)

def save_model(models):
    for pos, model in models.items():
        joblib.dump(model, f'{MODEL_PATH}_{pos}')
    