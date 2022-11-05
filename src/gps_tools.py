import copy
import csv
import json
import nvector as nv
import os

from tools import marks_google_dir
from math import radians, cos, sin, asin, sqrt


class MarkLocation:

    location: list[float]
    label: str

    def __init__(self, location: list[float], label: str):
        self.location = location
        self.label = label


def json_to_mark_location(filename):
    points = []
    with open(filename) as json_file:
        marks = json.load(json_file)
        for mark in marks["marks"]:
            points.append(
                MarkLocation(
                    [mark["position"]["latitude"], mark["position"]["longitude"]],
                    mark["label"],
                )
            )
    return points


def create_all_marks_cvs(mark_folder_name: str):
    '''
        Convert all marks in a folder to CSV format for Google Maps
        
        Parameters
        -----------
        
        mark_folder_name : name of the folder containing the marks
    '''
    for filename in os.listdir(mark_folder_name):
        if filename.endswith(".json"):
            convert_mark_json_to_csv(f"{mark_folder_name}/{filename}")

def convert_mark_json_to_csv(filename: str):
    points = json_to_mark_location(filename)
    convert_csv_gmaps(points, filename)


def convert_csv_gmaps(points: list[MarkLocation], output_name: str)-> None:
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


def interpolation(
    gps_location1: list[float], gps_location2: list[float], count: int
) -> list[list[float]]:
    """
    ## Interpolate between two GPS locations
    Given two GPS locations, this function will return a list of GPS locations that are interpolated between the two locations.
    The number of locations to be obtained by interpolation is passed as the count parameter.

    Parameters
    ----------
    gps_location1 : GPS location1
    gps_location2 : GPS location2
    count: number of locations to be obtained by interpolation
    """
    # create list of GPS locations
    new_points: list[list[float]] = []
    #  add location1 to list
    # new_points.append(gps_location1)

    # create nvector frame
    wgs84 = nv.FrameE(name="WGS84")
    # convert location1 to nvector
    n_EB_E_t0 = wgs84.GeoPoint(
        gps_location1[0], gps_location1[1], degrees=True
    ).to_nvector()
    # convert location2 to nvector
    n_EB_E_t1 = wgs84.GeoPoint(
        gps_location2[0], gps_location2[1], degrees=True
    ).to_nvector()
    # path between location1 and location2
    path = nv.GeoPath(n_EB_E_t0, n_EB_E_t1)
    t0 = 10
    t1 = t0 * (count + 2)

    for i in range(2, count + 2):
        ti = t0 * i  # time of interpolation

        ti_n = (ti - t0) / (t1 - t0)  # normalized time of interpolation
        # interpolate between location1 and location2 at time ti
        g_EB_E_ti = path.interpolate(ti_n).to_geo_point()
        # convert nvector to  latitude, longotude GPS location
        lat_ti, lon_ti, _ = g_EB_E_ti.latlon_deg
        # add interpolated location to list
        new_points.append([lat_ti, lon_ti])
    # add location2 to list
    # new_points.append(gps_location2)
    return new_points


def add_interpolate_location_to_samples(latitudesList, longitudesList):
    currentLocation = 0
    latCopy = copy.deepcopy(latitudesList)
    lntCopy = copy.deepcopy(longitudesList)
    for i in range(1, len(latitudesList)):
        if (
            latitudesList[i] != latitudesList[currentLocation]
            and longitudesList[i] != longitudesList[currentLocation]
            or (i + 1) == len(latitudesList)
        ):
            count = i - currentLocation - 1

            if count > 0:
                gps_location1 = [
                    latitudesList[currentLocation]
                    if (i + 1) != len(latitudesList)
                    else latitudesList[currentLocation - 1],
                    longitudesList[currentLocation]
                    if (i + 1) != len(latitudesList)
                    else latitudesList[currentLocation - 1],
                ]
                gps_location2 = [latitudesList[i], longitudesList[i]]
                new_interpolate_points = interpolation(
                    gps_location1, gps_location2, count
                )

                for index_i_points, [latitude, longitude] in enumerate(
                    new_interpolate_points
                ):
                    latCopy[currentLocation + 1 + index_i_points] = latitude
                    lntCopy[currentLocation + 1 + index_i_points] = longitude
                currentLocation = i

    return latCopy, lntCopy


# a = interpolation([23.13102, -82.36181], [23.13369, -82.36078], 10)
# convert_csv_gmaps(a)
# print(a)


def harvisine_distance(location1, location2, to_meters=False)->float:
    '''
        Distance between two points on earth using Harvisine  formula

        Parameters
        ----------
        location1 : GPS location1
        location2 : GPS location2
        to_meters: if True, the distance will be returned in meters

        Returns
        -------
        distance between location1 and location2

    '''
    # approximate radius of earth in km
    # Its equatorial radius is 6378 km, but its polar radius is 6357 km (WGS84)
    R = 6371.0
    # radians which converts from degrees to radians.
    lat1 = radians(location1[0])
    lon1 = radians(location1[1])
    lat2 = radians(location2[0])
    lon2 = radians(location2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))

    distance = R * c

    return distance * 1000 if to_meters else distance



# convert_mark_json_to_csv(
#     "./data/marks/TerminalTrenes-Ayesteran_marks.json")
