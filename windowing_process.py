import pandas as pd

def build_windows(time_series: pd.DataFrame, window_size=20, window_step=1):
    """
    Apply a sliding window (windowing) technique
    to a given time series

    time_series: the input time series to which apply the process
    window_size: the size of the sliding window
    step: the size of the step by which modify the bounds of the window
    returns: a set of pandas dataframes, each one representing a window
    within the time series

    """

    assert window_step < window_size

    lower_bound = 0
    upper_bound = window_size - 1
    time_series_size = len(time_series)

    windows = []
    while upper_bound < time_series_size:
        windows.append(pd.DataFrame(time_series[lower_bound:upper_bound]))
        lower_bound += window_step
        upper_bound += window_step
        
    return windows
