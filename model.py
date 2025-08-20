import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import joblib
from model_config import FEATURES, MODEL_PATH, MODEL_NAME, NUM_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT


def split_train_test(data):
    train_data = data[data['GW'] <= 30]
    test_data = data[data['GW'] > 30]
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
        pos_data = train_data[train_data['position'] == pos]
        model.fit(pos_data[FEATURES], pos_data['total_points'])
        models[pos] = model
    return models

def evaluate_model(models, test_data):
    all_model_predictions = []
    all_true_values = []
    # extra: initialize dictionary to collect feature coefficients
    feature_importances = {f: [] for f in FEATURES}
    for pos, model in models.items():
        pos_data = test_data[test_data['position'] == pos]
        predictions = model.predict(pos_data[FEATURES])
        all_model_predictions.extend(predictions)
        all_true_values.extend(pos_data['total_points'])
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
        'True Points': true_values,
        'Predicted Points': model_predictions
    })
    non_zero_df = results_df[results_df['True Points'] != 0].head(100)
    print("\nPREDICTED VS. TRUE POINTS")
    with pd.option_context('display.max_rows', 100, 'display.max_columns', None):
        print(non_zero_df)

def save_model(models):
    for pos, model in models.items():
        joblib.dump(model, f'{MODEL_PATH}_{pos}')
    