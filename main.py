from data_processing import get_data
from windowing_process import build_windows
from candt_anmly_heurs import find_candidates_heurs
from candt_anomaly_oults import find_candidates_outls

def main():
    time_seriess = get_data()
    idx = 0
    test_time_series = time_seriess[idx]

    windows = build_windows(test_time_series)

    heur_candts = find_candidates_heurs(test_time_series, idx)
    outls_candts = find_candidates_outls(test_time_series)

    print(heur_candts)
    print(outls_candts)
    # outl_candts = find_candidates_outl(test_time_series)

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
