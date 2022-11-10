import statistics

from named_dataframe import NamedDataframe
from sklearn.feature_selection import (
    SequentialFeatureSelector,
    RFE,
    RFECV,
    SelectFromModel
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

import pandas as pd

def add_features(ndt: NamedDataframe) -> NamedDataframe:
    '''
        Generate some extra features for the dataset to improve the 
        prediction performance.

        Parameters
        --------------

        ndt: Named data frame from where to generate the new features.

        Returns
        -------------

        A new data frame with the features added.

    '''

    dt = ndt.series
    dt['X / Z'] = dt['X Accel'] / dt['Z Accel']
    dt['MaxZratio'] = dt['Z Accel'] / max(dt['Z Accel'])
    dt['MinZratio'] = dt['Z Accel'] / min(dt['Z Accel'])
    dt['SpeedvsZ'] = dt['Speed'] / dt['Z Accel']

    # Statistic featurers
    dt['MeanDevAccelX'] = abs(dt['X Accel'] - statistics.mean(dt['X Accel']))
    dt['MedianDevAccelX'] = abs(dt['X Accel'] - statistics.median(dt['X Accel']))
    dt['MeanDevAccelY'] = abs(dt['Y Accel'] - statistics.mean(dt['Y Accel']))
    dt['MedianDevAccelY'] = abs(dt['Y Accel'] - statistics.median(dt['Y Accel']))
    dt['MeanDevAccelZ'] = abs(dt['Z Accel'] - statistics.mean(dt['Z Accel']))
    dt['MedianDevAccelZ'] = abs(dt['Z Accel'] - statistics.median(dt['Z Accel']))
    dt['MeanDevGyroX'] = abs(dt['X Gyro'] - statistics.mean(dt['X Gyro']))
    dt['MedianDevGyroX'] = abs(dt['X Gyro'] - statistics.median(dt['X Gyro']))
    dt['MeanDevGyroY'] = abs(dt['Y Gyro'] - statistics.mean(dt['Y Gyro']))
    dt['MedianDevGyroY'] = abs(dt['Y Gyro'] - statistics.median(dt['Y Gyro']))
    dt['MeanDevGyroZ'] = abs(dt['Z Gyro'] - statistics.mean(dt['Z Gyro']))
    dt['MedianDevGyroZ'] = abs(dt['Z Gyro'] - statistics.median(dt['Z Gyro']))
    return NamedDataframe(dt, ndt.id)

def feature_selection(X: pd.DataFrame, y: list[int], features_to_select: int):
    '''
        Select the best features for the model using several feature
        selection strategies.

        Parameters
        -----------

        X: The data to which apply the feature selection process.
        y: The classes of every data in the data set.
        features_to_select: The amount of features to select.

        Returns
        -----------

        A list of tuples representing the selector method name and the
        selected features with each one of them.

    '''

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    clsf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1)
    lrc = LogisticRegression()
    sfs_selector = SequentialFeatureSelector(estimator=clsf, n_features_to_select=features_to_select, direction='forward', n_jobs=-1)
    sbs_selector = SequentialFeatureSelector(estimator=clsf, n_features_to_select=features_to_select, direction='backward', n_jobs=-1)
    rfe_selector = RFE(estimator=clsf, n_features_to_select=features_to_select)
    rfe_cv_selector = RFECV(estimator=clsf, min_features_to_select=features_to_select, n_jobs=-1)
    sfm_selector = SelectFromModel(estimator=clsf, max_features=features_to_select)

    feature_selectors = [
        ('forward_selection', sfs_selector),
        ('backward_selection', sbs_selector),
        ('select_from_model_selection', sfm_selector),
        ('recursive_elimination', rfe_selector),
        ('cv_recursive_elimination', rfe_cv_selector)
    ]

    # cv_feature_selectors = [
    #     ('cv_recursive_elimination', rfe_cv_selector),
    # ]

    result = []
    # cv_features = []

    for selector_name, selector in feature_selectors:
        selector.fit(X, y)
        if selector_name == 'cv_recursive_elimination':
            selected = selector.support_
            all_features = X.columns
            features = []
            for i in range(len(selected)):
                if selected[i]:
                    features.append(all_features[i])
            result.append((selector_name, features))

        else:
            features = []
            for elem in X.columns[selector.get_support()]:
                features.append(elem) 
            result.append((selector_name, features))

    # for cv_selector_name, cv_selector in cv_feature_selectors:
    #     selector = cv_selector.fit(X, y)
    #     print(f"Features selected by {cv_selector_name} {}")

    return result

def remove_noise_features(time_series: pd.DataFrame):
    time_series = time_series.drop(['Latitude', 'Longitude', 'Accuracy'], axis=1)
    return time_series
