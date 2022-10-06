from os import listdir 
from os.path import isfile, join, splitext

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn import metrics  # for evaluations
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler  # for feature scaling

from tools import serialize_data, deserialize_data

def json_to_df() -> list[pd.DataFrame]:
    """
    Build dataframes from JSON files each one representing a time series
    """

    json_files = [
        (f"./data/{f_name}", splitext(f_name)[0]) for f_name in listdir("./data") if isfile(join("./data", f_name))
    ]

    procc_dfs = []

    for f_path, f_name in json_files:
        file = pd.read_json(f_path)

        accel_raw = file[["accelerometer"]].copy()
        speed_raw = file[["speed"]].copy()
        procc_data = []

        for index in range(len(accel_raw)):
            for list in accel_raw.iloc[index]:
                procc_data.append([elem for elem in list])

            procc_data[-1].append(speed_raw.iloc[index][0])

        procc_df = pd.DataFrame(
            procc_data, columns=["Accel X", "Accel Y", "Accel Z", "Speed"]
        )
        procc_dfs.append(procc_df)
        serialize_data(procc_df, f"./serialized_data/{f_name}")

    return procc_dfs

def main():

    procc_dfs = []
    for pickle_file in listdir("./serialized_data"):
        procc_dfs.append(deserialize_data(f"./serialized_data/{pickle_file}"))

    if not procc_dfs:
        procc_dfs = json_to_df()

    print(f"\n{procc_dfs[0]}")
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
