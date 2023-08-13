import argparse
import glob
import os

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main(args):
    mlflow.start_run()
    mlflow.sklearn.autolog()

    # read data
    df = read_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    accuracy = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

    # Log the metric with MLflow
    mlflow.log_metric("accuracy", accuracy)

    # end run
    mlflow.end_run()


def read_csvs_df(path):
    csv_files = glob.glob(f"{path}/*.csv")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df):
    """Splits data into training and testing sets."""
    y = df["label"]
    X = df.drop(columns=["label"])
    return train_test_split(X, y, test_size=0.3, random_state=0)


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(C=1/reg_rate)
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument(
        "--reg_rate", dest='reg_rate', type=float, default=0.01)
    # parse args
    args = parser.parse_args()
    # validate args
    if not os.path.exists(args.training_data):
        raise RuntimeError(
            f"Cannot use non-existent path provided: {args.training_data}")
    # return args
    return args


# run script
if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)

    args = parse_args()
    main(args)

    print("*" * 60)
    print("\n\n")
