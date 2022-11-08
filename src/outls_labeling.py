from os import listdir
import pandas as pd
import re

from tools import marks_google_dir
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
    for outl_idx in range(len(outls)):
        for mark_idx in range(len(marks)):
            mark_location = marks.iloc[mark_idx, "Location"]
            outl_location = (outls.iloc[outl_idx, "Latitude"], outls.iloc[outl_idx, "Longitude"])

            distance = haversine_distance(mark_location, outl_location)

            if distance < radius:
                classes.append(1)
                break

    return classes

def find_marks_file(series_name: str) -> pd.DataFrame:
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

    marks = []
    for f_name in listdir(marks_google_dir):
        main_name = f_name.split('_')[0]
        if re.match(marks_regex, main_name):
            marks = pd.read_csv(f"{marks_google_dir}/{marks_regex}.csv")
            break

    return marks
