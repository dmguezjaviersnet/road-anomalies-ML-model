from os import listdir
import pandas as pd
import re

from tools import marks_dir

def label_outls(outls: pd.DataFrame, series_name: str, radius=50) -> pd.DataFrame:
    """
    Asign a label to every outlier by searching the nearest mark using 
    haversine distance.

    outls: The set of outliers.
    series_name: The name of the series to choose the corresponding mark set.
    radius: The radius (in meters) within where to search for a mark to
    assign the label.

    """

    marks = find_marks_file(series_name)


def find_marks_file(series_name: str) -> pd.DataFrame:
    """
    Uses the time series identifier name to find the corresponding
    marks file.

    series_name: Name of the time series.

    """

    marks_regex = rf"{series_name}_marks"

    marks = []
    for f_name in listdir(marks_dir):
        main_name = f_name.split('_')[0]
        if re.match(marks_regex, main_name):
            marks = pd.read_csv(f"{marks_dir}/{marks_regex}.csv")
            break

    return marks


    

