from data_processing import fetch_data, marks_json_to_df
from outls_plots import outls_scatter
from outls_detection import detect_outls
from windowing_process import build_windows
from tools import samples_dir, marks_dir

# from IPython.display import display

def main():
    marks_dfs = marks_json_to_df(f"{marks_dir}")
    time_seriess = fetch_data(f"{samples_dir}")

    # //\\// ------------------ Taking outliers along the whole time series --------------------------//\\//

    for elem in time_seriess:
        predictions = detect_outls(elem.series)
        # outls_scatter(elem.series, predictions, rows=2, cols=3)

    # candt_windows = filter_candt_windows(windows, )

    # //\\// ------------------ Taking outliers by windows --------------------------//\\//

    # for elem in time_seriess:
    #     windows = build_windows(elem.series)
    # a = harvisine_distance([23.1300619, -82.3774041], [23.1294062, -82.3581093], True)
    a = harvisine_distance((23.1300619, -82.3774041), (23.1294062, -82.3581093), True)
    
    # print(a)


if __name__ == "__main__":
    main()
