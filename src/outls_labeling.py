from os import listdir
import pandas as pd
import re

from tools import marks_google_dir, str_to_tuple
from gps_tools import haversine_distance


def label_outls(outls: pd.DataFrame, series_name: str, radius=10) -> list[int]:
    '''
        Asign a label to every outlier by searching the nearest mark using 
        haversine distance.

        Parameters
        --------------

        outls: The set of outliers.
        series_name: The name of the series to choose the corresponding mark set.
        radius: The radius (in meters) within where to search for a mark to
        assign the label.

        Returns
        ---------------

        A list of ints representing the classes

    '''

    marks = find_marks_file(series_name)

    classes = [0]*len(outls)

    if not marks is None:
        for outl_idx in range(len(outls)):
            for mark_idx in range(len(marks)):
                mdf_location_index = marks.columns.get_loc("Location")
                mark_location = str_to_tuple(str(marks.iloc[mark_idx, mdf_location_index]))
                odf_latitude_index = outls.columns.get_loc("Latitude")
                odf_longitude_index = outls.columns.get_loc("Longitude")
                outl_location = (float(str(outls.iloc[outl_idx, odf_latitude_index])), float(str(outls.iloc[outl_idx, odf_longitude_index])))

                distance = haversine_distance(mark_location, outl_location, True)

                if distance < radius:
                    classes[outl_idx] = 1
                    break

    return classes


def find_marks_file(series_name: str) -> pd.DataFrame | None:
    '''
        Uses the time series identifier name to find the corresponding
        marks file.

        Parameters
        --------------

        series_name: Name of the time series.

        Returns
        --------------

        A pandas dataframe with the corresponding marks.

    '''

    marks_regex = rf"{series_name}_marks"

    for f_name in listdir(marks_google_dir):
        if re.match(marks_regex, f_name):
            return pd.read_csv(f"{marks_google_dir}/{marks_regex}.csv")

    return None
