import pandas as pd

from candt_anmly_heurs import find_candidates_heurs
from candt_anmly_unspv import find_candidates_unspv

def detect_outls(time_series: pd.DataFrame):
    """ 
        Detect outliers in a time series using several techniques such as 
        threshold-based heuristics and unsupervised learning methods

        Parameters
        ------------------

        :time_series: The time series to which apply the outlier detection process.

    """

    heur_candts, z_thresh_pred, z_diff_pred, g_zero_pred = find_candidates_heurs(time_series)
    dbscan_pred, optics_pred, ocsvm_pred = find_candidates_unspv(time_series)

    predictions = [("z_thresh", z_thresh_pred), ("z_diff", z_diff_pred), ("g_zero", g_zero_pred),
                    ("dbscan", dbscan_pred), ("optics", optics_pred), ("ocsvm", ocsvm_pred)]

    return predictions
