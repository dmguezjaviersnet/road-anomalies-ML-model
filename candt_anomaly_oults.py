import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from sklearn import metrics
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

def find_candidates_outls(time_series: pd.DataFrame):
    """
    Finds outliers in the time series as potential candidate anomalies using
    several algorithms.

    time_series: The time series to which apply the outliers detection
    algorithms.
    returns: A tuple consisting of the label(cluster) to which every data point
    belongs according to every algorithm. 

    """

    values = time_series[["X Accel",  "Y Accel", "Z Accel"]].values

    y_pred_dbscan = DBSCAN(eps=0.4, min_samples=30).fit_predict(values)
    y_pred_optics = OPTICS(min_samples=20, max_eps=.35).fit_predict(values)
    y_pred_ocsvm = OneClassSVM(kernel="rbf", gamma="scale").fit_predict(values)

    plt.subplot(131)
    plt.scatter(values[:, 0], values[:, 2], c=y_pred_dbscan)
    plt.title("DBSCAN")

    plt.subplot(132)
    plt.scatter(values[:, 0], values[:, 2], c=y_pred_optics)
    plt.title("OPTICS")

    plt.subplot(133)
    plt.scatter(values[:, 0], values[:, 2], c=y_pred_ocsvm)
    plt.title("OneClassSVM")

    plt.show()

    print('Mean Silhouette score DBSCAN: {}'.format(metrics.silhouette_score(values, y_pred_dbscan)))
    print('Mean Silhouette score OPTICS: {}'.format(metrics.silhouette_score(values, y_pred_optics)))
    print('Mean Silhouette score OneClassSVM: {}'.format(metrics.silhouette_score(values, y_pred_ocsvm)))

    return y_pred_ocsvm
