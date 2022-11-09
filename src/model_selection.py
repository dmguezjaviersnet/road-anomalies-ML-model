from sklearn import metrics
# from sklearn.utils import resample
# from sklearn.experimental import enable_halving_search_cv
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

def select_model(series_outls: pd.DataFrame, class_vector: list[int]):
    '''
        Train and evaluate several Machine Learning supervised methods to find the one
        that fits the best to this particular time series data-set.

        Parameters
        -----------------

        series_outl: The time series to use in the model selection process.
        class_vector: The vector with the corresponding classes.

    '''

    X_train, X_test, y_train, y_test = train_test_split(series_outls, class_vector, train_size=0.7)

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
            'n_estimators': [100, 120, 140, 160, 180],
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 4, 5, 6],
            'max_features': ['sqrt', 'log2', None]
        }
    ]

    logreg_param_grid = [
        {
            'penalty': ['l1', 'l2'],
            'tol': [1e-3, 1e-4, 1e-5, 1e-6],
            'C': [1, 10, 100, 1000],
            'solver': ['liblinear', 'sag', 'saga'],
            'max_iter': [100, 1000, 2000]
        }
    ]

    svm_param_grid = [
        {
            'C': [1, 10, 100, 1000],
            'kernel': ['linear']
        },

        {
            'C': [1, 10, 100, 1000],
            'kernel': ['poly', 'rbf', 'sigmoid'],
            'degree': [3, 4, 5, 6],
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
                ("Decision Tree", dt_clsf, dt_param_grid),
                ("Random Forest", rdf_clsf, rdf_param_grid),
                ("Log Regression", logreg_clsf, logreg_param_grid),
                ("SVM", svm_clsf, svm_param_grid)
    ]

    for clsf_name, clsf, clsf_param_grid in clsfrs:
        train_with_cv(clsf, clsf_param_grid, X_train, y_train)
        

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

    The mean f1 score(accuracy) and standard deviation. 

    '''

    cross_validator = RepeatedStratifiedKFold(n_splits=5, n_repeats=30, random_state=121)
    hyp_estm_cv = GridSearchCV(estimator=clsf, param_grid=param_grid, scoring=metrics.f1_score, cv=cross_validator)
    hyp_estm_cv.fit(X_train, y_train)

    results = pd.DataFrame(hyp_estm_cv.cv_results_)

    # fold = 1
    # f1_scores = []
    # for train_idx, val_idx in cross_validator.split(X, y):
    #     X_train = X.loc[train_idx]
    #     y_train = y.loc[train_idx]

    #     X_val = X.loc[val_idx]
    #     y_val = y.loc[val_idx]

    #     clsf.fit(X_train, y_train)
    #     pred = clsf.predict_proba(X_val)[:,1]
    #     f1_score = metrics.f1_score(y_val, pred)
    #     print(f"----------------  Fold {fold} --------------------")
    #     print( f"F1 score in validation set is {f1_score:0.4f}")
    #     fold += 1
    #     f1_scores.append(f1_score)

    # ovrall_acc = np.mean(f1_scores)
    # ovrall_std = np.std(f1_scores)
    # print(f"Mean F1 score is {ovrall_acc:0.4f}")
    # print(f"Standard deviation is {ovrall_std:0.4f}")