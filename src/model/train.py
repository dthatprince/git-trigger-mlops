# Import libraries
import argparse
import glob
import os
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# define functions
def main(args):
    mlflow.start_run()
    mlflow.sklearn.autolog()

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    accuracy = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

    # Log the metric with MLflow
    mlflow.log_metric("accuracy", accuracy)

    # end run
    mlflow.end_run()

def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

def split_data(df):
    """Splits data into training and testing sets."""
    if "label" not in df.columns:
        raise ValueError("The dataframe must have a 'label' column.")
    X = df.drop(columns=['label'])
    y = df['label']
    return train_test_split(X, y, test_size=0.3, random_state=0)

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
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
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)
    # parse args
    args = parser.parse_args()
    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
