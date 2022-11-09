import pandas as pd
import json
from os import listdir, path
from os.path import isfile, join, splitext, exists
import csv

from gps_tools import add_interpolate_location_to_samples
from named_dataframe import NamedDataframe
from tools import proc_samples_dir, marks_google_dir
from gps_tools import MarkLocation, add_interpolate_location_to_samples
import statistics


def json_samples_to_df(path: str) -> list[NamedDataframe]:
    '''
        Build dataframes from JSON files each one representing a time series.

        Parameters
        ----------------

        path: Path from where to read the JSONs.

        Returns
        -----------------

        A list of pandas dataframes containing each time series.

    '''

    json_files = [
        (f"{path}/{f_name}", splitext(f_name)[0]) for f_name in listdir(path) if isfile(join(path, f_name))
    ]

    named_dfs = []

    for f_path, f_name in json_files:
        # main_name = f_name.split('_')[0]
        if not exists(f"{proc_samples_dir}/{f_name}.csv"):
            print(f"File {f_name} is not serialized, collecting JSON...")

            with open(f_path) as json_file:
                data = json.load(json_file)

                time_series: pd.DataFrame = pd.json_normalize(data["records"])

                accel_raw = time_series[["accelerometer"]].copy()
                gyro_raw = time_series[["gyroscope"]].copy()
                speed_raw = time_series[["speed"]].copy()

                latitude_raw = time_series["gps.latitude"].copy()
                longitude_raw = time_series["gps.longitude"].copy()
                acc_raw = time_series["gps.accuracy"].copy()

                proc_data = []
                for index in range(len(accel_raw)):
                    accel_data: list[float] = accel_raw.iloc[index][0]
                    gyro_data: list[float] = gyro_raw.iloc[index][0]
                    motion_sensors_data = accel_data + gyro_data

                    speed_data = speed_raw.iloc[index][0]
                    lat_data = latitude_raw.iloc[index]
                    long_data = longitude_raw.iloc[index]
                    acc_data = acc_raw.iloc[index]
                    location_data = [lat_data, long_data, acc_data, speed_data]

                    data = motion_sensors_data + location_data
                    proc_data.append(data)

                proc_df = pd.DataFrame(
                    proc_data, columns=["X Accel", "Y Accel", "Z Accel",
                                        "X Gyro", "Y Gyro", "Z Gyro",
                                        "Latitude", "Longitude", "Accuracy", "Speed"]
                )

                latitudesList = proc_df["Latitude"].to_numpy()
                longitudesList = proc_df["Longitude"].to_numpy()
                proc_df["Latitude"], proc_df["Longitude"] = add_interpolate_location_to_samples(
                    latitudesList, longitudesList)

                # ///\\// ------ Delete all readings with less than 10m of accuracy -------- //\\//
                indexNames = proc_df[(proc_df['Accuracy'] > 10)].index
                proc_df.drop(indexNames, inplace=True)

                proc_df.to_csv(f"{proc_samples_dir}/{f_name}.csv", index=False)

                named_df = NamedDataframe(proc_df, f_name)
                named_dfs.append(named_df)
        else:
            series = pd.read_csv(f"{proc_samples_dir}/{f_name}.csv")
            named_df = NamedDataframe(series, f_name)
            named_dfs.append(named_df)

    return named_dfs


def convert_points_to_csv_gmaps_format(points: list[MarkLocation], output_name: str) -> None:
    '''
        ## Convert locations  to CSV format for Google Maps

        Parameters
        ----------

        points : list of locations given in the [latitude, longitude] format
        output_name : name of the output file  

    '''

    # csv header
    fieldnames = ["Name", "Location", "Accuracy", "Label"]
    rows = []
    # csv data
    for i, markLocation in enumerate(points):
        rows.append(
            {
                "Name": f"Point{i}",
                "Location": (markLocation.location[0], markLocation.location[1]),
                "Accuracy": f"{markLocation.accuracy}",
                "Label": f"{markLocation.label}",
            }
        )
    # write to csv
    with open(f'{marks_google_dir}/{path.basename(output_name).split(".")[0]}.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def export_df_to_csv(df: pd.DataFrame, output_name: str) -> None:
    df.to_csv(f"{output_name}.csv")

def mark_json_to_mark_location(filename: str) -> list[MarkLocation]:
    '''
        Convert a json file to a list of MarkLocation objects

        Parameters
        -----------
        filename: name of the json file

        Returns
        ----------

       A list of MarkLocation objects

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
                    (mark["position"]["latitude"], mark["position"]["longitude"]),
                    mark["position"]["accuracy"],
                    mark["label"],
                )
            )

    return points


def convert_mark_json_to_csv(filename: str) -> None:
    '''
        Convert a json file to a csv file for Google Maps

        Parameters
        -----------

        filename: name of the json file

    '''

    # convert json file to MarkLocation objects
    points = mark_json_to_mark_location(filename)
    # convert MarkLocation objects to csv format
    convert_points_to_csv_gmaps_format(points, filename)

def marks_json_to_df(path) -> list[NamedDataframe]:
    '''
        Convert all json marks  in a folder to CSV format for Google Maps

        Parameters
        -----------

        path : name of the folder containing the marks

        Returns
        -----------

        A list of NamedDataframe objects

    '''

    named_dfs = []

    #  for each file in the folder
    for filename in listdir(path):
        # if the file is a json file
        if filename.endswith(".json"):
            # if the file is not already converted to csv
            main_name = splitext(filename)[0]
            if not exists(f"{marks_google_dir}/{main_name}.csv"):
                # convert the json file to csv
                convert_mark_json_to_csv(f"{path}/{filename}")
                # read the csv file
                mark_df = pd.read_csv(f"{marks_google_dir}/{main_name}.csv")
                named_df = NamedDataframe(mark_df, main_name)
                named_dfs.append(named_df)

            else:
                # read the csv file
                mark_df = pd.read_csv(f"{marks_google_dir}/{main_name}.csv")
                named_df = NamedDataframe(mark_df, main_name)
                named_dfs.append(named_df)

    return named_dfs
