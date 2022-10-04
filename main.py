from os import listdir, wait, mkdir
from typing import Any
from os.path import isfile, join
from os import path
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles
from IPython.display import display
from sklearn import metrics  # for evaluations
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler  # for feature scaling

from tools import get_data

def make_pickle_file(file_name, data):
    with open(f"{file_name}.pickle", "wb") as outfile:
        pickle.dump(data, outfile)

def unpick_pickle_file(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    
    return data

def serialize_data(data, file_name: str):
    if not path.exists('./serialized_data'):
        mkdir('./serialized_data')

    make_pickle_file(file_name, data)

def deserialize_data(file_name):
    if path.exists(file_name):
        data = unpick_pickle_file(file_name)
        return data
    
    else: return None

def load_data_json():
    json_files = [
        f"./data/{file}" for file in listdir("./data") if isfile(join("./data", file))
    ]

    file1 = json_files[0]
    first = pd.read_json(file1)
    merged_raw_data = first

    for file in json_files[1:]:
        json_readed = pd.read_json(file)
        merged_raw_data = pd.concat([merged_raw_data, json_readed], axis=0)

    accel_raw = merged_raw_data[["accelerometer"]].copy()
    speed_raw = merged_raw_data[["speed"]].copy()

    data_procc = []

    for index in range(len(accel_raw)):
        for list in accel_raw.iloc[index]:
            data_procc.append([elem for elem in list])

        data_procc[-1].append(speed_raw.iloc[index][0])

    procc_df = pd.DataFrame(data_procc, columns=["Accel X", "Accel Y", "Accel Z", "Speed"])
    serialize_data(procc_df, './serialized_data/data_df')
    return procc_df

def main():
    procc_df = deserialize_data('./serialized_data/data_df.pickle')
    if not isinstance(procc_df, pd.DataFrame):
        procc_df = load_data_json()
    
    print(f"\n{procc_df}")
    # z_vs_x = procc_df[["Accel X", "Accel Z"]]
    # z_vs_x_thousand = procc_df.iloc[1:1000]
    # z_vs_x_thousand = z_vs_x_thousand[["Accel X", "Accel Y"]]
    
    # values = z_vs_x_thousand.values
    # values = StandardScaler().fit_transform(values)

    # y_pred = DBSCAN(eps=0.3, min_samples=30).fit_predict(values)
    # print(f"\n{y_pred}")
    # print(f"\n{values}")
    # plt.scatter(values[:, 0], values[:, -1], c=y_pred)

if __name__ == "__main__":
    main()
