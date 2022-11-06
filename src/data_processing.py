import pandas as pd
import json
from os import listdir
from os.path import isfile, join, splitext, exists

from gps_tools import add_interpolate_location_to_samples
from named_series import NamedSeries
from tools import proc_samples_dir, create_req_dirs


def json_samples_to_df(path: str) -> list[NamedSeries]:
import os
import csv
from named_dataframe import NamedDataframe
from gps_tools import MarkLocation, add_interpolate_location_to_samples


# -------------------- Required directories for the data ------------------------ #
data_dir = Path("./data")
csvs_dir = Path("./data/csvs")
marks_dir = Path("./data/marks")
samples_dir = Path("./data/samples")
proc_samples_dir = Path("./data/csvs/proc_samples")
marks_google_dir = Path("./data/csvs/marks")


def json_to_df(path: str) -> list[NamedDataframe]:
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

                label_col = ["No label"]*len(proc_data)
                proc_df["Label"] = label_col

                latitudesList = proc_df["Latitude"].to_numpy()
                longitudesList = proc_df["Longitude"].to_numpy()

                proc_df["Latitude"], proc_df["Longitude"] = add_interpolate_location_to_samples(
                    latitudesList, longitudesList)

                proc_df.to_csv(f"{proc_samples_dir}/{f_name}.csv")

                named_df = NamedDataframe(proc_df, main_name)
                named_dfs.append(named_df)

        else:
            series = pd.read_csv(f"{proc_samples_dir}/{f_name}.csv")
            named_df = NamedDataframe(series, main_name)
            named_dfs.append(named_df)

    return named_dfs



def convert_points_to_csv_gmaps_format(points: list[MarkLocation], output_name: str)-> None:
    '''
        ## Convert locations  to CSV format for Google Maps
        Parameters
        ----------
        points : list of locations given in the [latitude, longitude] format
        output_name : name of the output file  
    '''

    # csv header
    fieldnames = ["Name", "Location", "Description"]
    rows = []
    # csv data
    for i, markLocation in enumerate(points):
        rows.append(
            {
                "Name": f"Point{i}",
                "Location": (markLocation.location[0], markLocation.location[1]),
                "Description": f"{markLocation.label}",
            }
        )
    # write to csv
    with open(f'./data/csvs/marks/{os.path.basename(output_name).split(".")[0]}.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def mark_json_to_mark_location(filename):
    '''
        Convert a json file to a list of MarkLocation objects
    '''
    points = []
    # open json file
    with open(filename) as json_file:
        # load json file
        marks = json.load(json_file)
        # for each mark in the json file
        for mark in marks["marks"]:
            points.append(
                MarkLocation(
                    [mark["position"]["latitude"], mark["position"]["longitude"]],
                    mark["label"],
                )
            )
    return points


def convert_mark_json_to_csv(filename: str):
    points = mark_json_to_mark_location(filename)
    convert_points_to_csv_gmaps_format(points, filename)

def marks_json_to_df(path) -> list[NamedDataframe]:
    '''
        Convert all json marks  in a folder to CSV format for Google Maps

        Parameters
        -----------

        path : name of the folder containing the marks
    '''
    named_dfs = []

    #  for each file in the folder 
    for filename in os.listdir(path):
        # if the file is a json file
        if filename.endswith(".json"):
            # if the file is not already converted to csv
            if not exists(f"{marks_google_dir}/{filename}.csv"):
                # convert the json file to csv
                convert_mark_json_to_csv(f"{path}/{filename}")
                # read the csv file
                mark_df = pd.read_csv(f"{marks_google_dir}/{filename}.csv")
                named_df = NamedDataframe(mark_df, filename)
                named_dfs.append(named_df)

            else:
                # read the csv file
                mark_df = pd.read_csv(f"{marks_google_dir}/{filename}.csv")
                named_df = NamedDataframe(mark_df, filename)
                named_dfs.append(named_df)

    return named_dfs


def fetch_data(path: str) -> list[NamedDataframe]:
    """
        Get the pandas dataframes generated from raw JSON data in case
        it doesn't already exists. Otherwise just read the csv.

        Parameters
        ----------------

        path: Path from where to read the JSONs.

    """

    create_req_dirs()

    proc_dfs = []
    proc_dfs = json_samples_to_df(path)

    return proc_dfs
