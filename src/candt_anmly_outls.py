import pandas as pd

from sklearn import metrics
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.svm import OneClassSVM
# import matplotlib.cm as cm
# import numpy as np
# from IPython.display import display
# from sklearn.preprocessing import StandardScaler

def find_candidates_outls(time_series: pd.DataFrame) -> tuple:
    """
    Finds outliers in the time series as potential candidate anomalies using
    several algorithms.

    time_series: The time series to which apply the outliers detection
    algorithms.
    returns: A tuple consisting of the label(cluster) to which every data point
    belongs according to every algorithm. 

    """

    values = time_series[["X Accel", "Y Accel", "Z Accel"]].values

    y_pred_dbscan = DBSCAN(eps=0.4, min_samples=30).fit_predict(values)
    y_pred_optics = OPTICS(min_samples=20, max_eps=.35).fit_predict(values)
    y_pred_ocsvm = OneClassSVM(kernel="rbf", gamma="scale").fit_predict(values)

    print('Mean Silhouette score DBSCAN: {}'.format(metrics.silhouette_score(values, y_pred_dbscan)))
    print('Mean Silhouette score OPTICS: {}'.format(metrics.silhouette_score(values, y_pred_optics)))
    print('Mean Silhouette score OneClassSVM: {}'.format(metrics.silhouette_score(values, y_pred_ocsvm)))

    return y_pred_dbscan, y_pred_optics, y_pred_ocsvm