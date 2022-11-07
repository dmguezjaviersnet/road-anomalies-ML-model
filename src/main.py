from tabulate import tabulate
from IPython.display import display

from data_processing import marks_json_to_df, json_samples_to_df
from outls_plots import outls_scatter
from outls_detection import detect_outls
from windowing_process import build_windows
from tools import samples_dir, marks_dir, create_req_dirs


def main():
    create_req_dirs()
    marks_dfs = marks_json_to_df(f"{marks_dir}")
    time_seriess_df = json_samples_to_df(f"{samples_dir}")

    # //\\// ------------------ Taking outliers along the whole time series --------------------------//\\//

    # for elem in time_seriess_df:
    #     predictions = detect_outls(elem.series)
        # outls_scatter(elem.series, predictions, rows=2, cols=3)

    # print(tabulate(time_seriess_df[0].series, headers = 'keys', tablefmt = 'psql'))
    # candt_windows = filter_candt_windows(windows, )

    # //\\// ------------------ Taking outliers by windows --------------------------//\\//

    # for elem in time_seriess:
    #     windows = build_windows(elem.series)
    # a = harvisine_distance([23.1300619, -82.3774041], [23.1294062, -82.3581093], True)
    
    # print(a)


if __name__ == "__main__":
    main()
