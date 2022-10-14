from data_processing import get_data
from windowing_process import build_windows
from candt_anmly_heurs import find_candidates_heurs
from candt_anmly_outls import find_candidates_outls
from outls_plots import outls_scatter

def main():
    # //\\// ------------------ Testing scatter plot and heuristic predictions --------------------------//\\//
    time_seriess = get_data()

    for _, elem in enumerate(time_seriess):
        heur_candts, z_thresh_pred, z_diff_pred, g_zero_pred = find_candidates_heurs(elem)
        dbscan_pred, optics_pred, ocsvm_pred = find_candidates_outls(elem)

        predictions = [("z_thresh", z_thresh_pred), ("z_diff", z_diff_pred), ("g_zero", g_zero_pred),
                        ("dbscan", dbscan_pred), ("optics", optics_pred), ("ocsvm", ocsvm_pred)]

        outls_scatter(elem, predictions, rows=2, cols=3)
        print(heur_candts)
            
    # //\\// ------------------ Building windows --------------------------//\\//

    # windows = build_windows(test_time_series)
    # window_idx = 0
    # window = windows[window_idx]

if __name__ == "__main__":
    main()
