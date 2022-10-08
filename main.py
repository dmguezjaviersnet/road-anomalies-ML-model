from os import listdir 
from os.path import isfile, join, splitext

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from sklearn import metrics  # for evaluations
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler  # for feature scaling

from data_processing import get_data

def main():
    time_series = get_data()
    # z_vs_x = procc_df[["Accel X", "Accel Z"]]
    # z_vs_x_thousand = procc_df.iloc[1:1000]
    # z_vs_x_thousand = z_vs_x_thousand[["Accel X", "Accel Y"]]

    # values = z_vs_x_thousand.values
    # values = StandardScaler().fit_transform(values)

    # y_pred = DBSCAN(eps=0.3, min_samples=30).fit_predict(values)
    # print(f"\n{y_pred}")
    # print(f"\n{values}")
    # plt.scatter(values[:, 0], values[:, -1], c=y_pred)


if __name__ == "__main__":
    main()
