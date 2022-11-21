import json
import os
from pathlib import Path
from os.path import exists
import pandas as pd
import uuid
# //\\//  -------------------- Required directories for the data ----------------------- //\\\// #
data_dir = Path("./data")
csvs_dir = Path("./data/csvs")
marks_dir = Path("./data/marks")
samples_dir = Path("./data/samples")
serialized_preproce_data_dir = Path("./serialized_data")
proc_samples_dir = Path("./data/csvs/proc_samples")
marks_google_dir = Path("./data/csvs/marks")
test_csvs_dir = Path("./data/grid-search-results")
best_configs_dir = Path("./data/best-configs")
features_count_dir = Path("./data/feature-selected-count")
images_dir = Path("./data/images")

# Earth Gravity: 9.807 m/s²
g_e = 9.807
# Threshold fofr Z-THRESH HEURISTIC
z_thresh_threshold = round(0.25*g_e + g_e, 6)
# Threshold for Z-DIFF HEURISTIC
z_diff_threshold = round(7, 6)
# Threshold for G-ZERO HEURISTIC
g_zero_threshold = round(0.8*g_e, 6)

valid_model_names = ["knn", "dt", "rf", "logr", "svm"]

def create_req_dirs() -> None:
    '''
        Create the required directories in case they doesn't exist

    '''

    required_dirs = [data_dir, csvs_dir, marks_dir, samples_dir, proc_samples_dir,
                     marks_google_dir, test_csvs_dir, best_configs_dir, features_count_dir, serialized_preproce_data_dir, images_dir]

    for req_dir in required_dirs:
        if not exists(req_dir):
            req_dir.mkdir(parents=True, exist_ok=True)


def remove_parenthesis(s: str) -> str:
    '''
        Remove parenthesis from a string.

        Parameters
        ----------------

        s: String to remove parenthesis from.

        Returns
        ----------------

        String without parenthesis.

    '''

    return s.replace("(", "").replace(")", "")


def str_to_tuple(s: str) -> tuple:
    '''
        Convert a string to a tuple.

        Parameters
        ----------------

        s: String to convert to tuple.

        Returns
        ----------------

        Tuple with the values of the string.

    '''
    s = remove_parenthesis(s)
    return tuple(map(float, s.split(",")))


def remove_split_scores(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Remove the split scores from the dataframe.

        Parameters
        ----------------

        dt: Dataframe to remove the split scores from.

        Returns
        ----------------

        Dataframe without the split scores.

    '''

    regex = r"split\d+_test_score"
    return df[df.columns.drop(list(df.filter(regex=regex)))]


def save_to_json(new_data: dict, filename: str, key: str ="configs") -> None:
    '''
        Save a dataframe to a json file.

        Parameters
        ----------------

        df: Dataframe to save.
        filename: Name of the json file to save the dataframe to.

    '''
    if os.path.exists(filename):
        with open(filename, 'r+') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
            # Join new_data with file_data inside configs
            #  If file_data already has key, then append to it
            if key in file_data:
                file_data[key].append(new_data)
            else:
            # If file_data doesn't have key, then add it and new_datas
                file_data[key] = [new_data]
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file, indent=4)
    else:
        with open(filename, 'w') as file:
            json.dump({key: [new_data]}, file, indent=4)


def generate_unique_id_matrix(model: str):
    '''
        Generate a unique id for each model.

        Parameters
        ----------------

        model: Model to generate a unique id for.

        Returns
        ----------------

        Unique id for the model.

    '''
    return f"{uuid.uuid1()}-{model}"


def print_latex_knn_grid_search():
    df = pd.DataFrame()
    df['n_neighbors'] = ['3, 5, 7, 9, 11'],
    df['weights'] = ['uniform, distance']
    df['algorithm'] = ['brute, kd_tree, ball_tree']
    df['leaf_size'] = ['20, 30, 40, 50']

    return df.to_latex()


def print_latex_dt_grid_search():
    df = pd.DataFrame()
    df['criterion'] = ['entropy, gini'],
    df['splitter'] = ['best, random'],
    df['max_depth'] = ['3, 4, 5, 6, None'],
    df['max_features'] = ['sqrt, log2, None']

    return df.to_latex()


def print_latex_rf_grid_search():
    df = pd.DataFrame()
    df['n_estimators'] = ['100, 120, 180'],
    df['criterion'] = ['entropy, gini'],
    df['max_depth'] = ['10,  13, 16, 20'],
    df['max_features'] = ['log2, sqrt'],

    return df.to_latex()

def print_latex_reg_log_grid_search()-> str:
    df = pd.DataFrame()
    df['penalty'] = ['l2'],
    df['tol'] = ['1e-3, 1e-4, 1e-5, 1e-6'],
    df['C'] = ['1, 10, 100, 1000'],
    df['solver'] = ['lbfgs, saga'],
    df['max_iter'] = ['100, 500, 1000']
    return df.to_latex()

def print_latex_svm_config()-> str:
    df = pd.DataFrame()
    df['C']= ['100'],
    df['kernel']= ['rbf'],
    df['gamma']= ['0.01, 0.001'],
    df['probability']= [True]

    return df.to_latex()

print(print_latex_svm_config())
