from sklearn.metrics import f1_score, make_scorer
# from sklearn.utils import resample
# from sklearn.experimental import enable_halving_search_cv
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    # TimeSeriesSplit,
    # KFold,
    RepeatedStratifiedKFold,
    # StratifiedKFold,
    # HalvingGridSearchCV,
    GridSearchCV,
    # GroupKFold,
    # StratifiedGroupKFold,
    train_test_split,
    # cross_val_score
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tools import remove_split_scores

def select_model(series_outls: pd.DataFrame, class_vector: list[int]):
    '''
        Train and evaluate several Machine Learning supervised methods to find the one
        that fits the best to this particular time series data-set.

        Parameters
        -----------------

        series_outl: The time series to use in the model selection process.
        class_vector: The vector with the corresponding classes.

        Returns
        -----------------

        A list of tuples with the name of the classifier and the corresponding grdi 
        search results pandas dataframe.

    '''

    scaler = StandardScaler()
    series_outls_scaled = pd.DataFrame(scaler.fit_transform(series_outls), columns=series_outls.columns)

    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(series_outls_scaled, class_vector, train_size=0.6)
    X_train, X_test, y_train, y_test = train_test_split(series_outls, class_vector, train_size=0.6)

    knn_param_grid = [
        { 
            'n_neighbors' : [3, 5, 7, 9, 11],
            'weights' : ['uniform', 'distance'],
            'algorithm': ['brute'] 
        },

        {
            'n_neighbors' : [3, 5, 7, 9, 11],
            'weights' : ['uniform', 'distance'],
            'algorithm': ['kd_tree', 'ball_tree'],
            'leaf_size': [20, 30, 40, 50]
        }
    ]

    dt_param_grid = [
        {
            'criterion': ['entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [3, 4, 5, 6, None],
            'max_features': ['sqrt', 'log2', None]
        },
    ] 

    rdf_param_grid = [
        {
            # 'n_estimators': [100, 120, 140, 160, 180],
            'n_estimators': [100, 140, 180],
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 6, 9],
            'max_features': ['sqrt', 'log2']
        }
    ]

    logreg_param_grid = [
        {
            'penalty': ['l2'],
            #'tol': [1e-3, 1e-4, 1e-5, 1e-6],
           'tol': [1e-3, 1e-5],
            ##'C': [1, 10, 100, 1000],
            'C': [1, 100],
            'solver': ['lbfgs'],
            #'max_iter': [100, 500, 1000]
            'max_iter': [100, 500]
        },

        # {
        #     'penalty': ['l1'],
        #     'tol': [1e-3, 1e-4, 1e-5, 1e-6],
        #     'C': [1, 10, 100, 1000],
        #     'solver': ['saga'],
        #     'max_iter': [100, 500, 1000],
        #     'l1_ratio': [1]
        # }
    ]

    svm_param_grid = [
        #{
        #    #'C': [1, 10, 100, 1000],
        #    'C': [1, 100],
        #    'kernel': ['linear']
        #},

        {
            'C': [1, 10, 100],
            'kernel': ['sigmoids', 'rbf'],
            # 'kernel': ['rbf'],
            # 'degree': [3, 4],
            # 'gamma': [0.01, 0.001, 0.0001],
            'gamma': [0.01, 0.001, 0.0001],
            'probability': [True]
        }
    ]

    knn_clsf = KNeighborsClassifier()
    dt_clsf = DecisionTreeClassifier()
    rdf_clsf = RandomForestClassifier()
    logreg_clsf = LogisticRegression()
    svm_clsf = SVC()

    clsfrs = [
                ("KNN", knn_clsf, knn_param_grid),
                # ("Decision Tree", dt_clsf, dt_param_grid),
                # ("Random Forest", rdf_clsf, rdf_param_grid),
                # ("Log Regression", logreg_clsf, logreg_param_grid),
                # ("SVM", svm_clsf, svm_param_grid)
    ]

    results = []
    for clsf_name, clsf, clsf_param_grid in clsfrs:
        print(f"-------------------Running model {clsf_name}------------------------")
        if clsf_name == "Log Regression":
            clsf_gs_results = (clsf_name, train_with_cv(clsf, clsf_param_grid, X_train_scaled, y_train_scaled))         

        else:
            clsf_gs_results = (clsf_name, train_with_cv(clsf, clsf_param_grid, X_train, y_train))         

        results.append(clsf_gs_results)

    return results

def train_with_cv(clsf, param_grid, X_train: pd.DataFrame, y_train: list[int]):
    '''
    Train and evaluate the perfomance of a classifier using cross validation with 
    k-fold(RepeatedStratifiedKFold) whilst performing a grid search over a parameter 
    set to choose the best ones.

    Parameters
    ----------------

    clsf: The classifier to use.
    param_grid: The set of parameters to use in the grid search.
    X_train: Training data.
    y_train: Training data classes.

    Returns
    ----------------

    A dataframe with the Grid Search results.

    '''

    f1_scorer = make_scorer(f1_score)
    cross_validator = RepeatedStratifiedKFold(n_splits=5, n_repeats=30, random_state=121)
    hyp_estm_cv = GridSearchCV(estimator=clsf, param_grid=param_grid, scoring=f1_scorer, cv=cross_validator, n_jobs=-1)
    hyp_estm_cv.fit(X_train, y_train)

    gsearch_results = pd.DataFrame(hyp_estm_cv.cv_results_)

    return remove_split_scores(gsearch_results)
