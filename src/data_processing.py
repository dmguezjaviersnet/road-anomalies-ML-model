import pandas as pd
import json
from os import listdir, mkdir
from os.path import isfile, join, splitext, exists

from serializer import serialize_data, deserialize_data

def json_to_df() -> list[pd.DataFrame]:
    """
    Build dataframes from JSON files each one representing a time series

    """

    json_files = [
        (f"./data/{f_name}", splitext(f_name)[0]) for f_name in listdir("./data") if isfile(join("./data", f_name))
    ]

    proc_dfs = []

    for f_path, f_name in json_files:
        if not exists(f"./serialized_data/{f_name}.pickle"):
            print(f"File {f_name} does not exist, collecting JSON...")

            with open(f_path) as json_file:
                data = json.load(json_file)
                time_series = pd.json_normalize(data["records"])

                accel_raw = time_series[["accelerometer"]].copy()
                speed_raw = time_series[["speed"]].copy()
                proc_data = []

                for index in range(len(accel_raw)):
                    for list in accel_raw.iloc[index]:
                        proc_data.append([elem for elem in list])

                    proc_data[-1].append(speed_raw.iloc[index][0])

                proc_df = pd.DataFrame(
                    proc_data, columns=["X Accel", "Y Accel", "Z Accel", "Speed"]
                )
                proc_dfs.append(proc_df)
                serialize_data(proc_df, f"./serialized_data/{f_name}")

        else:
            proc_dfs.append(deserialize_data(f"./serialized_data/{f_name}"))

    return proc_dfs

def get_data() -> list[pd.DataFrame]:
    """
    Get the pandas dataframes generated from raw JSON data in case
    it doesn't already exists. Otherwise just deserialize it.

    """

    proc_dfs = []
    if not exists("./serialized_data"):
        mkdir("./serialized_data")

    if len(listdir("./serialized_data")) < len(listdir("./data")):
        print("Missing files, checking directory")
        proc_dfs = json_to_df()

    else: 
        for pickle_file in listdir("./serialized_data"):
            proc_dfs.append(deserialize_data(f"./serialized_data/{pickle_file}"))

    return proc_dfs
