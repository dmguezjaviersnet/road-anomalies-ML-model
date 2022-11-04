import pandas as pd
import json
from os import listdir, mkdir
from os.path import isfile, join, splitext, exists
from gps_tools import add_interpolate_location_to_samples

from serializer import serialize_data, deserialize_data
from named_series import NamedSeries


# from re import match


def json_to_df(path: str) -> list[NamedSeries]:
    """
    Build dataframes from JSON files each one representing a time series.

    Parameters
    --------------------

    path: Path from where to read the JSONs.

    """

    json_files = [
        (f"{path}/{f_name}", splitext(f_name)[0]) for f_name in listdir(path) if isfile(join(path, f_name))
    ]

    named_dfs = []

    for f_path, f_name in json_files:
        if not exists(f"./serialized_data/{f_name}.pickle"):
            print(f"File {f_name} is not serialized, collecting JSON...")

            with open(f_path) as json_file:
                data = json.load(json_file)

                time_series = pd.json_normalize(data["records"])
                print(time_series.info())

                accel_raw = time_series[["accelerometer"]].copy()
                speed_raw = time_series[["speed"]].copy()
                latitude_raw = time_series["gps.latitude"].copy()
                longitude_raw = time_series["gps.longitude"].copy()
                proc_data = []

                for index in range(len(accel_raw)):
                    proc_data.append(accel_raw.iloc[index][0])
                    proc_data[-1].append(speed_raw.iloc[index][0])
                    print(len(latitude_raw))
                    proc_data[-1].append(latitude_raw.iloc[index])
                    proc_data[-1].append(longitude_raw.iloc[index])

                proc_df = pd.DataFrame(
                    proc_data, columns=["X Accel", "Y Accel",
                                        "Z Accel", "Speed", "Latitude", "Longitude"]
                )
                latitudesList = proc_df["Latitude"].to_numpy()
                longitudesList = proc_df["Longitude"].to_numpy()

                proc_df["Latitude"], proc_df["Longitude"] = add_interpolate_location_to_samples(
                    latitudesList, longitudesList)

                print(proc_df)
                # serialize_data(proc_df, f"./serialized_data/{f_name}")

                named_df = NamedSeries(proc_df, f_name)
                named_dfs.append(named_df)
                serialize_data(named_df, f"./serialized_data/{f_name}")

        else:
            named_dfs.append(deserialize_data(f"./serialized_data/{f_name}"))

    return named_dfs


def get_data(path: str) -> list[NamedSeries]:
    """
    Get the pandas dataframes generated from raw JSON data in case
    it doesn't already exists. Otherwise just deserialize it.

    Parameters
    ----------------

    path: Path from where to read the JSONs.

    """

    proc_dfs = []
    if not exists("./serialized_data"):
        mkdir("./serialized_data")

    if len(listdir("./serialized_data")) < len(listdir(path)):
        print("Missing files, checking directory")
        proc_dfs = json_to_df(path)

    else:
        for pickle_file in listdir("./serialized_data"):
            proc_dfs.append(deserialize_data(
                f"./serialized_data/{pickle_file}"))

    return proc_dfs
