import pandas as pd
import json
from pathlib import Path
from os import listdir
from os.path import isfile, join, splitext, exists
from gps_tools import add_interpolate_location_to_samples

from named_series import NamedSeries

# -------------------- Required directories for the data ------------------------ #
data_dir = Path("./data")
csvs_dir = Path("./data/csvs")
marks_dir = Path("./data/marks")
samples_dir = Path("./data/samples")
proc_samples_dir = Path("./data/csvs/proc_samples") 
marks_google_dir = Path("./data/csvs/marks")

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
        main_name = f_name.split('_')[0]
        if not exists(f"{proc_samples_dir}/{f_name}.csv"):
            print(f"File {f_name} is not serialized, collecting JSON...")

            with open(f_path) as json_file:
                data = json.load(json_file)

                time_series = pd.json_normalize(data["records"])

                accel_raw = time_series[["accelerometer"]].copy()
                speed_raw = time_series[["speed"]].copy()
                latitude_raw = time_series["gps.latitude"].copy()
                longitude_raw = time_series["gps.longitude"].copy()
                proc_data = []

                for index in range(len(accel_raw)):
                    proc_data.append(accel_raw.iloc[index][0])
                    proc_data[-1].append(speed_raw.iloc[index][0])
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

                proc_df.to_csv(f"{proc_samples_dir}/{f_name}.csv")

                named_df = NamedSeries(proc_df, main_name)
                named_dfs.append(named_df)

        else:
            series = pd.read_csv(f"{proc_samples_dir}/{f_name}.csv")
            named_df = NamedSeries(series, main_name)
            named_dfs.append(named_df)

    return named_dfs


def fetch_data(path: str) -> list[NamedSeries]:
    """
    Get the pandas dataframes generated from raw JSON data in case
    it doesn't already exists. Otherwise just read the csv.

    Parameters
    ----------------

    path: Path from where to read the JSONs.

    """

    create_req_dirs()

    proc_dfs = []
    proc_dfs = json_to_df(path)

    return proc_dfs

def create_req_dirs() -> None:
    """
    Create the required directories in case they doesn't exist

    """

    required_dirs = [data_dir, csvs_dir, marks_dir, samples_dir, proc_samples_dir, marks_google_dir]
    
    for req_dir in required_dirs:
        if not exists(req_dir):
            req_dir.mkdir(parents=True, exist_ok=True)
