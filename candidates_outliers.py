import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from sklearn import metrics
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

def find_candidates_outl(time_series: pd.DataFrame):
    """
    Finds outliers in the time series as potential candidates anomalies using
    several algorithms.

    time_series: The time series to which apply the outliers detection
    algorithms.
    returns: A tuple consisting of the label(cluster) to which every data point
    belongs according to every algorithm. 

    """

    values = time_series.values

    # y_pred_dbscan = DBSCAN(eps=0.3, min_samples=30).fit_predict(values)
    # y_pred_optics = OPTICS(min_samples=20, max_eps=.35).fit_predict(values)
    y_pred_ocsvm = OneClassSVM(kernel="rbf", gamma="scale").fit_predict(values)

    # plt.scatter(values[:, 0], values[:, -1], c=y_pred_dbscan)
    # print('Mean Silhouette score: {}'.format(metrics.silhouette_score(values, y_pred_dbscan)))

    # plt.scatter(values[:, 0], values[:, -1], c=y_pred_optics)
    # print('Mean Silhouette score: {}'.format(metrics.silhouette_score(values, y_pred_optics)))

    plt.scatter(values[:, 0], values[:, -1], c=y_pred_ocsvm)
    print('Mean Silhouette score: {}'.format(metrics.silhouette_score(values, y_pred_ocsvm)))

    return y_pred_ocsvm
