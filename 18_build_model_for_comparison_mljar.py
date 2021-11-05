import argparse

import pandas as pd
from supervised.automl import AutoML
import pickle

folder = "./data/final_datasets/"
metric = "mae"
time_budget = 2 * 3600
excluded_columns = ["id", "breath_id", "pressure"]
target = "pressure"

def get_rc():
    parser = argparse.ArgumentParser(description='Collect tweets')
    parser.add_argument('--rc', type=str,  help='Number of seconds before to stop the collecter', default="20__50")
    args = parser.parse_args()
    rc = args.rc
    return rc

if __name__ == '__main__':

    rc = get_rc()

    print("Get data")
    dfp_train = pd.read_csv(folder + "train_final_2.csv")
    dfp_train_subpopulation = dfp_train.copy()
    features = [column for column in dfp_train_subpopulation.columns.tolist() if column not in excluded_columns]
    print("Features:", features)
    print(f"Size training:{len(dfp_train_subpopulation)}")
    X_train, y_train = dfp_train_subpopulation[features], dfp_train_subpopulation[target]
    del dfp_train

    print("Build model with autoML")
    automl = AutoML(mode="Compete", eval_metric="mae", results_path=f"./data/ensemble_approach/mljar/mljar_final_2", total_time_limit=time_budget)
    automl.fit(X_train, y_train)
    print("DONE")