from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    # TimeSeriesSplit,
    # KFold,
    StratifiedKFold,
    # GroupKFold,
    # StratifiedGroupKFold,
    train_test_split,
    cross_val_score
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

def select_model(series_outls: pd.DataFrame, class_vector: list[int]):
    '''
        Train and evaluate several Machine Learning supervised methods to find the one
        that fits the best to this particular time series data-set.

        Parameters
        -----------------

        series_outl: The time series to use in the model selection process.
        class_vector: The vector with the corresponding classes.

    '''

    X_train, X_test, y_train, y_test = train_test_split(series_outls, class_vector, train_size=0.8)

    nb_clsf = GaussianNB()
    knn_clsf = KNeighborsClassifier()
    dt_clsf = DecisionTreeClassifier(criterion="entropy")
    rdf_clsf = RandomForestClassifier()
    log_reg_clsf = LogisticRegression(solver='liblinear')
    svm_clsf = SVC(probability=True)

    clsfrs = [nb_clsf, knn_clsf, dt_clsf, rdf_clsf, log_reg_clsf, svm_clsf]

def train_with_cv(clsf , series_outls: pd.DataFrame, class_vector: pd.DataFrame, cv=StratifiedKFold):
    '''
    Train and evaluate the perfomance of a classifier using cross validation with 
    k-fold(StratifiedKFold by default).

    Parameters
    ----------------

    clsf: The classifier to use.
    series_outls: The time series to use as data.
    class_vector`: The prediction for each value in the time series.

    Returns
    ----------------

    The mean f1 score(accuracy) and standard deviation. 

    '''

    sk_fold = cv(n_splits=5, shuffle=True, random_state=238)
    X, y = series_outls, class_vector

    fold = 1
    f1_scores = []
    for train_idx, val_idx in sk_fold.split(X, y):
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]

        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]

        clsf.fit(X_train, y_train)
        pred = clsf.predict_proba(X_val)[:,1]
        f1_score = metrics.f1_score(y_val, pred)
        print(f"----------------  Fold {fold} --------------------")
        print( f"F1 score in validation set is {f1_score:0.4f}")
        fold += 1
        f1_scores.append(f1_score)

    ovrall_acc = np.mean(f1_scores)
    ovrall_std = np.std(f1_scores)
    print(f"Mean F1 score is {ovrall_acc:0.4f}")
    print(f"Standard deviation is {ovrall_std:0.4f}")

