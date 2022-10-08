import pandas as pd

def g_zero(accel_x, accel_y, accel_z, threshold=0.5) -> bool:
    """
    Threshold-based heuristic to determine the presence of
    an anomaly on the road using accelerometer reading from
    the 3 axis

    accel_x: x axis acceleration
    accel_y: y axis acceleration
    accel_z: z axis acceleration
    threshold: threshold to decide whether the readings represent
    an anomaly or not
    """

    thresh = 10;

    return (abs(accel_x) < thresh and
    abs(accel_y) < thresh and
    abs(accel_z) < thresh)

def z_thresh(accel_z, threshold=12) -> bool:
    """
    Threshold-based heuristic to determine the presence of an anomaly on the road
    using accelerometer reading from the z axis

    accel_z: z axis acceleration
    threshold: threshold to decide whether the readings represent
    an anomaly or not
    """

    return abs(accel_z) > threshold;

def z_diff(prev_accel_z, curr_accel_z, threshold=10) -> bool:
    """
    Threshold-based heuristic to determine the presence of an anomaly on the road
    using two consecutive accelerometer readings from the z axis

    prev_accel_z: previous z axis acceleration
    curr_accel_z: current z axis acceleration
    threshold: threshold to decide whether the readings represent
    an anomaly or not
    """

    return abs((curr_accel_z - prev_accel_z)) > threshold;

def std_dev_z(window: pd.DataFrame, threshold=15) -> bool:
    """
    Threshold-based heuristic to determine the presence of an anomaly on the road
    using the standard deviation of z axis accelerometer readings in a window of the time
    series

    accel_z: z axis acceleration
    threshold: threshold to decide whether the readings represent
    an anomaly or not
    """

    return window["Accel Z"].std() > threshold
