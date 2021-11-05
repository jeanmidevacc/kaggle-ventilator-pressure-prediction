import argparse

import pandas as pd
from supervised.automl import AutoML
import pickle

folder = "./data/final_datasets/"
metric = "mae"
time_budget = 2*3600
excluded_columns = ["id", "breath_id", "R__C", "R", "C", "pressure"]
rcs = ['20__50', '20__20', '50__20', '50__50', '5__50', '5__20', '50__10', '20__10', '5__10']
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
    dfp_train = pd.read_csv(folder + "train_final.csv")
    dfp_train_subpopulation = dfp_train[dfp_train["R__C"] == rc]
    features = [column for column in dfp_train_subpopulation.columns.tolist() if column not in excluded_columns]
    print("Features:", features)
    print(f"Size training:{len(dfp_train_subpopulation)}")
    X_train, y_train = dfp_train_subpopulation[features], dfp_train_subpopulation[target]
    del dfp_train

    print("Build model with autoML")
    automl = AutoML(mode="Compete", eval_metric="mae", results_path=f"./data/ensemble_approach/mljar/mljar{time_budget}_{rc}", total_time_limit=time_budget)
    automl.fit(X_train, y_train)
    print("DONE")