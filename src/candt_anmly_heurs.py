import pandas as pd

def g_zero(window: pd.DataFrame, threshold=0.5) -> bool:
    """
    Threshold-based heuristic to determine the presence of
    an anomaly on the road using accelerometer readings from
    the 3 axis

    window: window where to look for anomalies
    threshold: threshold to decide whether the readings represent
    an anomaly or not

    """

    for _, data_point in window.iterrows():
        if (data_point["X Accel"] < threshold and data_point["Y Accel"] < threshold
            and data_point["Z Accel"] < threshold):
            return True

    return False


def z_thresh(window: pd.DataFrame, threshold=12) -> bool:
    """
    Threshold-based heuristic to determine the presence of an anomaly on the road
    using accelerometer reading from the z axis

    window: window where to look for anomalies
    threshold: threshold to decide whether the readings represent
    an anomaly or not

    """

    z_accel = window["Z Accel"].tolist()

    return any(abs(data_point) > threshold for data_point in z_accel)

def z_diff(window: pd.DataFrame, threshold=10) -> bool:
    """
    Threshold-based heuristic to determine the presence of an anomaly on the road
    using two consecutive accelerometer readings from the z axis

    window: window where to look for anomalies
    threshold: threshold to decide whether the readings represent
    an anomaly or not

    """

    z_accel = window["Z Accel"].tolist()

    return any(abs(z_accel[index + 1] - z_accel[index]) > threshold for index, _ in enumerate(z_accel))

def std_dev_z(window: pd.DataFrame, threshold=15) -> bool:
    """
    Threshold-based heuristic to determine the presence of an anomaly on the road
    using the standard deviation of z axis accelerometer readings in a window of the time
    series

    window: window where to look for anomalies
    threshold: threshold to decide whether the readings represent
    an anomaly or not

    """

    return abs(window["Z Accel"].std()) > threshold

def find_candidates_heurs(window: pd.DataFrame, window_idx: int) -> dict[str, list[int]]:
    """
    Decide using each one of the 4 heuristics, which window 
    is a potential anomaly.

    windows: The windows which were generated with the windowing process
    window_idx: Index of the window in the time series.
    returns: A dictionary identifying which heuristic identified which windows
    (by index) as potential anomalies.

    """

    candidates = {"z-thresh": [], "z-diff": [], "g-zero": [], "stdev(z)": []}

    if z_thresh(window):
        candidates["z-thresh"].append(window_idx)

    if z_diff(window):
        candidates["z-diff"].append(window_idx)

    if g_zero(window):
        candidates["g-zero"].append(window_idx)

    if std_dev_z(window):
        candidates["stdev(z)"].append(window_idx)

    return candidates
