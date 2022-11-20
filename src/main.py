from unittest import result
from tabulate import tabulate
from IPython.display import display

from data_processing import export_df_to_csv, marks_json_to_df, json_samples_to_df
from features_processing import add_features, feature_selection, remove_noise_features
from outls_labeling import label_outls
from outls_plots import outls_scatter
from outls_detection import detect_outls, filter_outliers
from windowing_process import build_windows
from tools import samples_dir, marks_dir, create_req_dirs, save_to_json, best_configs_dir
from model_selection import select_model
import pandas as pd


def main():
    create_req_dirs()
    marks_json_to_df(f"{marks_dir}")
    time_seriess_df = json_samples_to_df(f"{samples_dir}")

    time_seriess_df_w_nf = [add_features(elem) for elem in time_seriess_df]
    # export_df_to_csv(with_features.series, "temporal")
    # //\\// ------------------ Taking outliers along the whole time series --------------------------//\\//

    for elem in time_seriess_df_w_nf:
        predictions = detect_outls(elem.series)
        # outls_scatter(elem.series, predictions, rows=2, cols=3)

        # export_df_to_csv(elem.series, f"{elem.id}_doble")
        outl_methods_sel = [elem for elem in predictions if elem[0] == "z_thresh" or elem[0] == "dbscan"]

        output_csv_idx = 1
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
                    features_selected_sets = feature_selection(X, y, 6)
                    print(f"---------------- Features selected with outliers detected using {outl_method_name} ----------------\n")

                    for selector_name, features_selected_set in features_selected_sets:
                        print(f"Features selected with selector {selector_name}")
                        print(f"{features_selected_set}\n")

                        df_sel_feats: pd.DataFrame = pd.DataFrame(X[features_selected_set])
                        ms_results = select_model(df_sel_feats, y)

                        for model_name, result in ms_results:
                            print(f"----- Results with {model_name} -----")
                            print(f"{result}\n\n")
                            config = {
                                "outliers_method": outl_method_name,
                                "feature_selector": selector_name,
                                "features_selected_set": features_selected_set,
                                "model": model_name,
                                "best-config": result["params"].values[:1][0],
                                "f1_score" : result["mean_test_score"].values[:1][0]
                            }
                            save_to_json(config, f"{best_configs_dir}/configs.json")
                            export_df_to_csv(result, f"{model_name}-Results-{output_csv_idx}")

                        output_csv_idx += 1
    # x = {
    #     "go": 45,
    #     "pepito": "P",
    #     "juanito": ["D", "A", "J"]
    # }

    # save_to_json(x, f"{best_configs_dir}/configs.json")


if __name__ == "__main__":
    main()
