import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def outls_scatter(time_series: pd.DataFrame, pred_sets: list[tuple[str, np.ndarray]], rows: int, cols: int) -> None:
    """
        Build a scatterplot for each output of each outlier prediction
        methods as subplots of a unique graph.
        
        Parameters
        -------------------

        predictions: A list containing tuples string, ndarray identifying
        the method and representing the prediction repectively
        rows: number of plot rows
        cols: number of plot cols

        rows*cols must be greater or equal than len(prediction).

    """

    graph_amount = len(pred_sets)
    assert graph_amount <= rows * cols

    values = time_series[["X Accel",  "Y Accel", "Z Accel"]].values

    for graph_index in range(graph_amount): 
        method_name, y_pred = pred_sets[graph_index]

        plt.subplot(rows, cols, graph_index + 1)
        plt.scatter(values[:, 0], values[:, 2], c=y_pred)
        plt.title(method_name)

        graph_index += 1

    plt.show()
