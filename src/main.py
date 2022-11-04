from data_processing import get_data
from outls_plots import outls_scatter
from outls_detection import detect_outls

def main():
    # //\\// ------------------ Testing scatter plot and heuristic predictions --------------------------//\\//
    time_seriess = get_data("./data/samples")

    for _, elem in enumerate(time_seriess):
        predictions = detect_outls(elem.series)
        outls_scatter(elem.series, predictions, rows=2, cols=3)
            
    # //\\// ------------------ Building windows --------------------------//\\//

    # windows = build_windows(test_time_series)
    # window_idx = 0
    # window = windows[window_idx]

if __name__ == "__main__":
    main()
