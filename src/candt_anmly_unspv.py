import pandas as pd

from sklearn import metrics
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.svm import OneClassSVM

# import matplotlib.cm as cm
# import numpy as np
# from sklearn.preprocessing import StandardScaler

def find_candidates_unspv(time_series: pd.DataFrame) -> tuple:
    '''
        Finds outliers in the time series as potential candidate anomalies using
        several algorithms.

        Parameters
        -----------------

        time_series: The time series to which apply the outliers detection
        algorithms.
        returns: A tuple consisting of the label(cluster) to which every data point
        belongs according to every algorithm. 

    '''

    values = time_series[["X Accel", "Y Accel", "Z Accel", "X Gyro", "Y Gyro", "Z Gyro"]].values

    y_pred_dbscan = DBSCAN(eps=0.6, min_samples=30).fit_predict(values)
    y_pred_optics = OPTICS(min_samples=3).fit_predict(values)
    y_pred_ocsvm = OneClassSVM(kernel="rbf", gamma="scale").fit_predict(values)

    # Silhouette Metric
    score_dbscan = metrics.silhouette_score(values, y_pred_dbscan)
    score_optics = metrics.silhouette_score(values, y_pred_optics)
    score_ocsvm = metrics.silhouette_score(values, y_pred_ocsvm)
    print('Silhouette score DBSCAN: {}'.format(score_dbscan))
    print('Silhouette score OPTICS: {}'.format(score_optics))
    print('Silhouette score OneClassSVM: {}'.format(score_ocsvm))

    return y_pred_dbscan, y_pred_optics, y_pred_ocsvm, score_dbscan, score_optics, score_ocsvm
