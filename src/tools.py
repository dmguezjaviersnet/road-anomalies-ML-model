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
z_diff_threshold = 4
# Threshold for G-ZERO HEURISTIC
# round(0.8*g_e, 6)
g_zero_threshold = 0.8

valid_model_names = ["knn", "dt", "rf", "logr", "svm"]

# ---Best hyperparams confs for clustering algs---
# DBSCAN 
best_dbscan_eps: float = 0.99
best_dbscan_min_samples:int  = 15
# OCSVM
best_ocsvm_gamma: float = 1e-05 
best_ocsvm_nu: float = 0.05
# OPTICS
best_optics_min_samples: int = 15
best_optics_method: str = 'xi'
best_optics_metric1: str = 'canberra'
best_optics_metric2: str = 'braycurtis'

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


def save_to_json(new_data: dict, filename: str, swap: bool, key: str ="configs") -> None:
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
                if swap:
                    data_config = file_data[key][0]
                    if new_data["f1_score_test"] > data_config["f1_score_test"]:
                        file_data[key] = [new_data]
                else:
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

def print_silhouette_route_results()-> str:
    df =  pd.DataFrame()
    df['Ruta'] = ['Ruta1', 'Ruta2', 'Ruta3']
    df['DBSCAN'] = ['0.502849', '0.659878', '0.620526']
    df['OCSVM']  = ['0.533472', '0.533707', '0.509438']
    df['OPTICS'] = ['-0.245268', '-0.271357', '-0.238291']
    return df.to_latex()

def print_silhouette_mean_results()-> str:
    df = pd.DataFrame()
    df['Mean'] = ['0.594417', '0.525539', '-0.251639']
    df['Algoritmo'] = ['DBSCAN', 'OCSVM', 'OPTICS']
    return df.to_latex()

def print_dbscan_hparams_conf()-> str:
    df = pd.DataFrame()
    df['eps'] = ['0.05, 0.1, 0.15, 0.20, ... 0.90, 0.95, 0.99']
    df['min_samples'] = ['15, 20, 25, 30, 25, 40,  45, 50, 55, 60, 65']
    return df.to_latex()

def print_ocsvm_hparams_conf()-> str:
    df = pd.DataFrame()
    df['gamma'] = ['scale, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10']
    df['nu'] = ['0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99']
    return df.to_latex()

def print_optics_hparams_conf()-> str:
    df = pd.DataFrame()
    df['cluster_method'] = ["xi, dbscan"]
    df['metric'] = ["minkowski, euclidean, canberra, braycurtis"]
    df['min_samples'] = ['5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65']
    return df.to_latex()

def print_outliers_detected()-> str:
    df = pd.DataFrame()
    df['Algoritmo'] = ['DBSCAN', 'OCSVM', 'z-diff', 'z-tresh', 'g_zero']
    df['Número de Anomalías'] = [134, 446, 181, 95, 0]
    return df.to_latex()



def print_latex_result(model_result: dict):
    df = pd.DataFrame()
    keys = []
    values = []
    for key in model_result.keys():
        if (key !='f1_score_train' and key !="confusion_matrix_path"
            and key != "n_outliers"):
            
            if key == 'best-config':
                temp = ""
                config_keys = []
                config_values = []
                for k, v in model_result[key].items():
                    temp += f"{k}: {v}, "
                    config_keys.append(k)
                    config_values.append(v)
                df2 = pd.DataFrame()
                df2['Hyperparams'] = config_keys
                df2['Value'] = config_values
                print("------Configuración de hiperparámetros del modelo------")
                print(df2.to_latex(index=False))
            else:
                if key == "f1_score_test":
                    keys.append("F1_score")
                elif key == "precision_test":
                    keys.append("Precision")
                elif key == "recall_test":
                    keys.append("Recall")
                elif key == "accuracy_test":
                    keys.append("Accuracy")
                else:
                    keys.append(key)
                values.append(model_result[key])
    df['Keys'] = keys
    df['Values'] = values
    return df.to_latex(index=False)



print("------Resultados------")
print(print_latex_result(
{
            "outliers_method": "ocsvm",
            "feature_selector": "forward_selection",
            "features_selected_set": [
                "X Gyro",
                "Y Gyro",
                "Z Gyro",
                "X / Z",
                "MeanDevAccelY",
                "MeanDevAccelZ",
                "MedianDevGyroY",
                "MeanDevGyroZ",
                "MedianDevGyroZ"
            ],
            "model": "Random Forest",
            "best-config": {
                "criterion": "gini",
                "max_depth": 20,
                "max_features": "sqrt",
                "n_estimators": 120
            },
            "f1_score_train": 0.25875019331037635,
            "f1_score_test": 0.2926829268292683,
            "precision_test": 0.75,
            "recall_test": 0.18181818181818182,
            "accuracy_test": 0.7835820895522388,
            "confusion_matrix_path": "data/images/1e34c2a8-6ac5-11ed-a6b2-bbf57c35a274-Random Forest.png",
            "n_outliers": 446
        }
))