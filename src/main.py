from unittest import result
from tabulate import tabulate
from IPython.display import display

from data_processing import export_df_to_csv, marks_json_to_df, json_samples_to_df
from features_processing import add_features, feature_selection, remove_noise_features
from outls_labeling import label_outls
from outls_plots import outls_scatter
from outls_detection import detect_outls, filter_outliers
from windowing_process import build_windows
from tools import samples_dir, marks_dir, create_req_dirs
from model_selection import select_model
import pandas as pd


def main():
    create_req_dirs()
    marks_dfs = marks_json_to_df(f"{marks_dir}")
    time_seriess_df = json_samples_to_df(f"{samples_dir}")
    
    time_seriess_df_w_nf = [add_features(elem) for elem in time_seriess_df]
    # export_df_to_csv(with_features.series, "temporal")
    # //\\// ------------------ Taking outliers along the whole time series --------------------------//\\//

    for elem in time_seriess_df_w_nf:
        predictions = detect_outls(elem.series)
        # outls_scatter(elem.series, predictions, rows=2, cols=3)

        # export_df_to_csv(elem.series, f"{elem.id}_doble")
        outl_methods_sel = [elem for elem in predictions if elem[0] == "z_thresh" or elem[0] == "dbscan"]
        f_index = 1
        for outl_method_name, outl_pred in outl_methods_sel:
            outliers = filter_outliers(elem.series, outl_pred)
            if len(outliers):
                # export_df_to_csv(outliers, f"{elem.id}_outliers")
                y = label_outls(outliers, elem.id, 10)
                if (not all(data_label == 0 for data_label in y) and
                    not all(data_label == 1 for data_label in y)):

                    outliers = remove_noise_features(outliers)
        
                    X = outliers
                    # X['EsBache'] = y
                    # export_df_to_csv(X, f"labeled_outliers")
                    features_selected_sets = feature_selection(X, y, 3)
                    print(f"---------- Features selected with outliers detected using {outl_method_name} ----------\n")
                    
                    
                    for selector_name, features_selected_set in features_selected_sets:
                        print(f"Features selected with selector {selector_name}")
                        print(f"{features_selected_set}\n")

                        df_sel_feats: pd.DataFrame = pd.DataFrame(X[features_selected_set]) 
                        ms_results = select_model(df_sel_feats, y)
                        
                        for model_name, result in ms_results:
                            print(f"-------- Resultados de {model_name}---------")
                            print(result)
                            export_df_to_csv(result, f"{model_name}-Results-{f_index}")

                        f_index += 1
                        

    # print(tabulate(time_seriess_df[0].series, headers = 'keys', tablefmt = 'psql'))
    # candt_windows = filter_candt_windows(windows, )

    # //\\// ------------------ Taking outliers by windows --------------------------//\\//

    # for elem in time_seriess:
    #     windows = build_windows(elem.series)
    # a = harvisine_distance([23.1300619, -82.3774041], [23.1294062, -82.3581093], True)
    #print(tabulate(with_features.series, headers = 'keys', tablefmt = 'psql'))
    #convert_mark_json_to_csv('data/marks/Ruta5TerminalTrenes-Ayesteran_y_19mayo_marks.json')
    # print(a)

if __name__ == "__main__":
    main()
