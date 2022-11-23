import pandas as pd

from sklearn import metrics
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.svm import OneClassSVM
from tools import best_dbscan_eps, best_dbscan_min_samples, best_ocsvm_gamma, best_ocsvm_nu, best_optics_min_samples, best_optics_method, best_optics_metric1, best_optics_metric2

# import matplotlib.cm as cm
# import numpy as np
# from sklearn.preprocessing import StandardScaler

def find_candidates_unspv(time_series: pd.DataFrame, route_name: str) -> tuple:
    '''
        Finds outliers in the time series as potential candidate anomalies using
        several algorithms.

        Parameters
        -----------------

        time_series: The time series to which apply the outliers detection
        algorithms.
        route_name: The name of the route to which the time series belongs.
        returns: A tuple consisting of the label(cluster) to which every data point
        belongs according to every algorithm. 

    '''

    values = time_series[["X Accel", "Y Accel", "Z Accel", "X Gyro", "Y Gyro", "Z Gyro"]].values
    
    #y_pred_dbscan = DBSCAN(eps=0.6, min_samples=30).fit_predict(values)
    #y_pred_optics = OPTICS(min_samples=3).fit_predict(values)
    
    #y_pred_ocsvm = OneClassSVM(kernel="rbf", gamma="scale").fit_predict(values)

    y_pred_dbscan = DBSCAN(eps=best_dbscan_eps, min_samples=best_dbscan_min_samples).fit_predict(values)
    y_pred_ocsvm = OneClassSVM(kernel="rbf", gamma=best_ocsvm_gamma, nu=best_ocsvm_nu).fit_predict(values)
    y_pred_optics = OPTICS(min_samples=best_optics_min_samples, cluster_method=best_optics_method, metric=best_optics_metric1).fit_predict(values)
    if len(y_pred_optics) > 1 and all(elem == y_pred_optics[0] for elem in y_pred_optics):
        y_pred_optics = OPTICS(min_samples=10, cluster_method=best_optics_method, metric=best_optics_metric2).fit_predict(values)
    # Silhouette Metric
    # best_hparams_dbscan = search_best_dbscan_hparams(values)
    #best_hparams_ocsvm = search_bebst_ocsvm_hparams(values)
    #best_hparams_optics = search_best_optics_hparams(values)
    #y_pred_optics = OPTICS(min_samples=best_hparams_optics["min_samples"], cluster_method=best_hparams_optics["cluster_method"], 
    #metric=best_hparams_optics["metric"]).fit_predict(values)
   

    # print(f"best eps dbscacn: {best_hparams_dbscan['eps']}")
    # print(f"best min_samples: {best_hparams_dbscan['min_samples']}")
    #y_pred_dbscan = DBSCAN(eps=best_hparams_dbscan['eps'], min_samples=best_hparams_dbscan['min_samples']).fit_predict(values)
    # Silhoutter score for each clustering method used
    score_dbscan = metrics.silhouette_score(values, y_pred_dbscan)
    score_ocsvm = metrics.silhouette_score(values, y_pred_ocsvm)
    score_optics = metrics.silhouette_score(values, y_pred_optics)
    #score_optics = metrics.silhouette_score(values, y_pred_optics)
    #score_ocsvm = metrics.silhouette_score(values, y_pred_ocsvm)
    print(f"--------------Route: {route_name}--------------------")
    print('Silhouette score DBSCAN: {}'.format(score_dbscan))
    print('Silhouette score OneClassSVM: {}'.format(score_ocsvm))
    print("Silhouette score OPTICS: {}".format(score_optics))
    print(f"-----------------------------------------------------")
    # print("Best eps: {}".format(best_hparams_dbscan['eps']))
    # print("Best min_samples: {}".format(best_hparams_dbscan['min_samples']))

    #y_pred_ocsvm = OneClassSVM(kernel="rbf", gamma=best_hparams_ocsvm['gamma']).fit_predict(values)
    #y_pred_ocsvm = OneClassSVM(kernel="rbf", gamma=best_hparams_ocsvm['gamma'], nu=best_hparams_ocsvm['nu']).fit_predict(values)
    
    # print("Best min_samples: {}".format(best_hparams_optics['min_samples']))
    # print("Best cluster_method: {}".format(best_hparams_optics['cluster_method']))
    # print("Best metric: {}".format(best_hparams_optics['metric']))
    #print('Silhouette score OneClassSVM: {}'.format(score_ocsvm))
    #print("Best gamma: {}".format(best_hparams_ocsvm['gamma']))
    #print("Best Nu: {}".format(best_hparams_ocsvm['nu']))
    #print('Silhouette score OPTICS: {}'.format(score_optics))
    #print('Silhouette score OneClassSVM: {}'.format(score_ocsvm))

    #return y_pred_dbscan, y_pred_optics, y_pred_ocsvm, score_dbscan, score_optics, score_ocsvm
    return y_pred_dbscan, y_pred_optics, y_pred_ocsvm, score_dbscan, score_optics, score_ocsvm 

def search_best_dbscan_hparams(values):
    epsilon_values = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.60 , 0.65, 0.70, 0.75,  0.80, 0.85, 0.90, 0.95, 0.99]
    min_samples_values = [15, 20, 25, 30, 35, 40,  45, 50, 55, 60, 65]
    best_hparams = {}
    for eps in epsilon_values:
        for min_samples in min_samples_values:
            y_pred_dbscan =  DBSCAN(eps=eps, min_samples=min_samples).fit_predict(values)
            if len(y_pred_dbscan) > 1 and all(elem == y_pred_dbscan[0] for elem in y_pred_dbscan):
                continue

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
    return best_hparams

def search_bebst_ocsvm_hparams(values):
    gammas_values =  ["scale", 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    nu_values= [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    best_hparams = {}
    for nu in nu_values:
        for g in gammas_values:
            y_pred_ocsvm = OneClassSVM(kernel="rbf", gamma=g, nu=nu).fit_predict(values)
            if len(y_pred_ocsvm) > 1 and all(elem == y_pred_ocsvm[0] for elem in y_pred_ocsvm):
                continue

            score_dbscan = metrics.silhouette_score(values, y_pred_ocsvm)
            config =  {
                "gamma": g,
                "nu": nu,
                "score" : score_dbscan
            }
            if best_hparams:
                best_hparams =  config if  score_dbscan > best_hparams['score'] else best_hparams
            else:
                best_hparams = config
    return best_hparams

def search_best_optics_hparams(values):
    #y_pred_optics = OPTICS(min_samples=3).fit_predict(values)  
    cluster_method_values = ["xi", "dbscan"]
    metrics_values = ["minkowski", "euclidean", "canberra", "braycurtis"] 
    min_samples_values =  [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    best_hparams = {}
    for cluster_method in cluster_method_values:
        for metric in metrics_values:
            for min_samples in min_samples_values:
                y_pred_optics = OPTICS(min_samples=min_samples, cluster_method=cluster_method, metric=metric, n_jobs=4).fit_predict(values)
                if len(y_pred_optics) > 1 and all(elem == y_pred_optics[0] for elem in y_pred_optics):
                    continue

                score_optics = metrics.silhouette_score(values, y_pred_optics)
                config =  {
                    "cluster_method": cluster_method,
                    "metric": metric,
                    "min_samples": min_samples,
                    "score" : score_optics
                }
                if best_hparams:
                    best_hparams =  config if  score_optics > best_hparams['score'] else best_hparams
                else:
                    best_hparams = config
    # db = OPTICS(max_eps=epsilon, min_samples=min_samples, cluster_method=cluster_method, metric=metric).fit(X)  
    return best_hparams

