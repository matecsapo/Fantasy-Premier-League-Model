import pandas as pd
from data_processing import load_data
from features import add_features
from model import split_train_test, train_model, evaluate_model, save_model
from model_config import FEATURES, NUM_ESTIMATORS
import argparse

def main():
    # model specification command line args
    parser = argparse.ArgumentParser(description="Run FPL prediction model")
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        choices=["linear", "randomforest"],
        help="Which model to use"
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default = NUM_ESTIMATORS,
        help="Number of trees (if using RandomForest)"
    )
    args = parser.parse_args()

    # load data
    data = load_data()

    # add features
    data = add_features(data)

    # label = total_points

    # drop any missing data
    data = data.dropna(subset=FEATURES)

    # split data into training and testing data
    train_data, test_data = split_train_test(data)

    # train model
    models = train_model(train_data, model_type = args.model, num_estimators = args.estimators)

    # evaluate model
    evaluate_model(models, test_data)

    # save model
    save_model(models)

if __name__ == "__main__":
    main()
