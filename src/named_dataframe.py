import pandas as pd

class NamedDataframe(object):
    """
    An object representing a time series and its identifier name based
    on the JSON file from where it was readed.

    Parameters
    --------------------

    series: The pandas dataframe representing the time series.
    id: The identifier name of the series.

    """

    def __init__(self, series: pd.DataFrame, id: str):
        self.series = series
        self.id = id
