import pandas as pd

from candt_anmly_heurs import find_candidates_heurs
from candt_anmly_unspv import find_candidates_unspv

def detect_outls(time_series: pd.DataFrame):
    '''
        Detect outliers in a time series using several techniques such as 
        threshold-based heuristics and unsupervised learning methods

        Parameters
        ------------------

        :time_series: The time series to which apply the outlier detection process.

    '''

    heur_candts, z_thresh_pred, z_diff_pred, g_zero_pred = find_candidates_heurs(time_series)
    dbscan_pred, optics_pred, ocsvm_pred = find_candidates_unspv(time_series)

    predictions = [("z_thresh", z_thresh_pred), ("z_diff", z_diff_pred), ("g_zero", g_zero_pred),
                    ("dbscan", dbscan_pred), ("optics", optics_pred), ("ocsvm", ocsvm_pred)]

    return predictions

def filter_outliers(time_series: pd.DataFrame, predictions: list):
    '''
        Get the outliers from the predictions of the outlier detection process 

        Parameters
        ------------------

        :time_series: time series dataframe
        :predictions: list of predictions from the outlier detection process

    '''

    time_series["Outlier"] = predictions
    indexes_to_remove = time_series[(time_series['Outlier'] != -1)].index
    time_series = time_series.drop(indexes_to_remove, axis=0)
    time_series = time_series.drop(['Outlier'], axis=1)
    return time_series
