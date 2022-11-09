import statistics
from named_dataframe import NamedDataframe
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd

def add_features(ndt: NamedDataframe)-> NamedDataframe:
    '''
    Generate some extra features for the data set to improve the 
    prediction accuracy.

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
    # Statistics featurers
    dt['MeanDevX'] = (dt['X Accel'] - statistics.mean(dt['X Accel']))**2
    dt['MedianDevX'] = (dt['X Accel'] - statistics.median(dt['X Accel']))**2 
    dt['MeanDevY'] = (dt['Y Accel'] - statistics.mean(dt['Y Accel']))**2
    dt['MedianDevY'] = (dt['Y Accel'] - statistics.median(dt['Y Accel']))**2 
    dt['MeanDevZ'] = (dt['Z Accel'] - statistics.mean(dt['Z Accel']))**2
    dt['MedianDevZ'] = (dt['Z Accel'] - statistics.median(dt['Z Accel']))**2 

    
    return NamedDataframe(dt, ndt.id)

def feature_selection_sfs(X, y, direction = 'backward'):
    '''
        Select the best features for the model using Sequential Feature Selection
        Parameters
        ----------
        X: The features of the data set
        y: The target of the data set
        direction: The direction of the selection, forward or backward
    '''
    # Selecting the Best important features according to Logistic Regression
    print(X.head())
    scaler = preprocessing.StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
    print(X.head())
    sfs_selector = SequentialFeatureSelector(estimator=LogisticRegression(), n_features_to_select = 10, cv =10, direction =direction)
    sfs_selector.fit(X, y)
    return X.columns[sfs_selector.get_support()]