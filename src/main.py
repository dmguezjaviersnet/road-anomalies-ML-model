from data_processing import fetch_data
from outls_plots import outls_scatter
from outls_detection import detect_outls
from windowing_process import build_windows
from gps_tools import create_all_marks_csv
from tools import samples_dir, marks_dir

# from IPython.display import display

def main():
    create_all_marks_csv(f"{marks_dir}")

    # //\\// ------------------ Taking outliers along the whole time series --------------------------//\\//

    time_seriess = fetch_data(f"{samples_dir}")

    for elem in time_seriess:
        predictions = detect_outls(elem.series)
        # outls_scatter(elem.series, predictions, rows=2, cols=3)

    # candt_windows = filter_candt_windows(windows, )

    # //\\// ------------------ Taking outliers by windows --------------------------//\\//

    # for elem in time_seriess:
    #     windows = build_windows(elem.series)

if __name__ == "__main__":
    main()
