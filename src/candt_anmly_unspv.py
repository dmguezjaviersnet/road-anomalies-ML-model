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
    best_hparams = search_best_dbscan_hparams(values)
    print(f"best eps dbscacn: {best_hparams['eps']}")
    print(f"best min_samples: {best_hparams['min_samples']}")
    score_dbscan = metrics.silhouette_score(values, y_pred_dbscan)
    score_optics = metrics.silhouette_score(values, y_pred_optics)
    score_ocsvm = metrics.silhouette_score(values, y_pred_ocsvm)
    print('Silhouette score DBSCAN: {}'.format(score_dbscan))
    print('Silhouette score OPTICS: {}'.format(score_optics))
    print('Silhouette score OneClassSVM: {}'.format(score_ocsvm))

    return y_pred_dbscan, y_pred_optics, y_pred_ocsvm, score_dbscan, score_optics, score_ocsvm


def search_best_dbscan_hparams(values):
    epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ,0.7 , 0.8, 0.9, 0.10]
    min_samples_values = [15, 30, 50, 70]
    best_hparams = {}
    for eps in epsilon_values:
        for min_samples in min_samples_values:
            y_pred_dbscan =  DBSCAN(eps=epsilon_values, min_samples=min_samples_values).fit_predict(values)
            score_dbscan = metrics.silhouette_score(values, y_pred_dbscan)
            config =  {
                    "eps": eps,
                    "min_samples": min_samples,
                    "score" : score_dbscan
            }
            if best_hparams:
               best_hparams =  config if  score_dbscan > best_hparams['score'] else best_hparams
            else:
                best_hparams = config

def search_best_ocsvm_hparams(values):
    # 0.001, 0.01, 0.1, 1, 10, 100 gammas 
    # 0.001, 0.01, 0.1, 1, 10, 100 C
    ...