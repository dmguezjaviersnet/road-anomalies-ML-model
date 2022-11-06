from data_processing import fetch_data
from outls_plots import outls_scatter
from outls_detection import detect_outls
from windowing_process import build_windows, filter_candt_windows
from gps_tools import harvisine_distance

# from IPython.display import display

def main():
    # //\\// ------------------ Testing scatter plot and heuristic predictions --------------------------//\\//
    # time_seriess = fetch_data("./data/samples")

    # for elem in time_seriess:
    #     windows = build_windows(elem.series)
    #     predictions = detect_outls(elem.series)
        # outls_scatter(elem.series, predictions, rows=2, cols=3)

    # candt_windows = filter_candt_windows(windows, )

    # //\\// ------------------ Building windows --------------------------//\\//

    a = harvisine_distance([23.1300619, -82.3774041], [23.1294062, -82.3581093], True)
    
    print(a)


if __name__ == "__main__":
    main()
