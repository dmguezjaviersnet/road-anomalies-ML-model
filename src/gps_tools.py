import copy
import json
import nvector as nv
from math import radians, cos, sin, asin, sqrt
import numpy as np


class MarkLocation:

    location: list[float]
    label: str

    def __init__(self, location: list[float], label: str):
        self.location = location
        self.label = label



def interpolation(
    gps_location1: tuple[float, float], gps_location2: tuple[float, float], count: int
) -> list[tuple[float, float]]:
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
    new_points: list[tuple[float, float]] = []
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
        new_points.append((lat_ti, lon_ti))
    # add location2 to list
    # new_points.append(gps_location2)
    return new_points


def add_interpolate_location_to_samples(latitudesList: np.ndarray, longitudesList: np.ndarray):
    currentLocation = 0
    latCopy = copy.deepcopy(latitudesList)
    lntCopy = copy.deepcopy(longitudesList)    # 
    for i in range(1, len(latitudesList)):
        if (
            latitudesList[i] != latitudesList[currentLocation]
            and longitudesList[i] != longitudesList[currentLocation]
            or (i + 1) == len(latitudesList)
        ):
            count = i - currentLocation - 1

            if count > 0:
                gps_location1 = (
                    latitudesList[currentLocation]
                    if (i + 1) != len(latitudesList)
                    else latitudesList[currentLocation - 1],
                    longitudesList[currentLocation]
                    if (i + 1) != len(latitudesList)
                    else latitudesList[currentLocation - 1],
                )
                gps_location2 = (latitudesList[i], longitudesList[i])
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



def harvisine_distance(location1: tuple[float, float], location2: tuple[float, float], to_meters: bool = False)->float:
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



