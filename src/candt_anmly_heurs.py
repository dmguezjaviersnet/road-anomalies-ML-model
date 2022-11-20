import pandas as pd
import numpy as np
from tools import z_thresh_threshold, z_diff_threshold, g_zero_threshold

def g_zero(window: pd.DataFrame, threshold=g_zero_threshold) -> tuple[bool, np.ndarray]:
    '''
        Threshold-based heuristic to determine the presence of
        an anomaly on the road using accelerometer readings from
        the 3 axis.

        Parameters
        --------------------

        window: window where to look for anomalies
        threshold: threshold to decide whether the readings represent
        an anomaly or not

    '''

    anmly_prsnce = False
    anmlies = [1]*len(window)

    for index, data_point in window.iterrows():
        if (abs(data_point["X Accel"]) < threshold and abs(data_point["Y Accel"]) < threshold
            and abs(data_point["Z Accel"]) < threshold):

            anmlies[index] = -1
            anmly_prsnce = True
    
    return anmly_prsnce, np.array(anmlies)


def z_thresh(window: pd.DataFrame, threshold=z_thresh_threshold) -> tuple[bool, np.ndarray]:
    '''
        Threshold-based heuristic to determine the presence of an anomaly on the road
        using accelerometer reading from the z axis.

        Parameters
        -------------------

        window: window where to look for anomalies
        threshold: threshold to decide whether the readings represent
        an anomaly or not

    '''

    anmly_prsnce = False
    anmlies = [1]*len(window)

    z_accel = window["Z Accel"].tolist()

    for index, data_point in enumerate(z_accel):
        if abs(data_point) > threshold:
            anmlies[index] = -1
            anmly_prsnce = True
    
    return anmly_prsnce, np.array(anmlies)

def z_diff(window: pd.DataFrame, threshold=z_diff_threshold) -> tuple[bool, np.ndarray]:
    '''
        Threshold-based heuristic to determine the presence of an anomaly on the road
        using two consecutive accelerometer readings from the z axis
        
        Parameters
        ----------------

        window: window where to look for anomalies
        threshold: threshold to decide whether the readings represent
        an anomaly or not

    '''

    anmly_prsnce = False
    anmlies = [1]*len(window)

    z_accel = window["Z Accel"].tolist()
    data_length = len(z_accel)

    for index, _ in enumerate(z_accel):
        if index < data_length - 1:
            if (abs(z_accel[index + 1] - z_accel[index]) > threshold):
                anmlies[index] = -1
                anmlies[index + 1] = -1
                anmly_prsnce = True
    
    return anmly_prsnce, np.array(anmlies)

def std_dev_z(window: pd.DataFrame, threshold=5) -> bool:
    '''
        Threshold-based heuristic to determine the presence of an anomaly on the road
        using the standard deviation of z axis accelerometer readings in a window of the time
        series
        
        Parameters
        --------------------

        window: window where to look for anomalies
        threshold: threshold to decide whether the readings represent
        an anomaly or not

    '''

    return abs(window["Z Accel"].std()) > threshold

def find_candidates_heurs(window: pd.DataFrame) -> tuple[list[bool], np.ndarray, np.ndarray, np.ndarray]:
    '''
        Decide using each one of the 4 heuristics, which window 
        contains potential anomalies.

        Parameters
        ------------------
        
        window: The window generated with the windowing process
        returns: A list identifying which heuristic identified
        potential anomalies in the window. 0: z-thresh, 1: z-diff,
        2: g-zero, 3: stdev(z), and in case of the first 3 methods also 
        returns a list with the exact data points identified as potential
        anomalies in the window

    '''

    candidates = [False]*4
    anmly = False

    anmly, z_thresh_anmlies = z_thresh(window)
    if anmly: candidates[0] = True

    anmly, z_diff_anmlies = z_diff(window)
    if anmly: candidates[1] = True

    anmly, g_zero_anmlies = g_zero(window)
    if anmly: candidates[2] = True

    anmly = std_dev_z(window)
    if anmly: candidates[3] = True

    return candidates, z_thresh_anmlies, z_diff_anmlies, g_zero_anmlies
