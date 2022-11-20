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
    mean_dbscan_score = 0
    mean_optics_score = 0
    mean_ocsvm_score =  0 

    outliers_df: dict[str, tuple[pd.DataFrame, list]] = {
        "z_thresh": (pd.DataFrame(), []),
        "z_diff": (pd.DataFrame(), []),
        "g_zero": (pd.DataFrame(), []),
        "dbscan": (pd.DataFrame(), []),
        "optics": (pd.DataFrame(), []),
        "ocsvm": (pd.DataFrame(), []),
    }

    for elem in time_seriess_df_w_nf:
        predictions, score_dbscan, score_optics, score_ocsvm = detect_outls(elem.series)
        mean_dbscan_score += score_dbscan
        mean_ocsvm_score  += score_ocsvm
        mean_optics_score += score_optics
        # outl_methods_sel = [elem for elem in predictions if elem[0] == "z_thresh" or elem[0] == "dbscan"]
        
        for outl_method_name in predictions.keys():
            outliers = filter_outliers(
                elem.series, predictions[outl_method_name])
            # if there is any outliers
            if len(outliers):
                # export_df_to_csv(outliers, f"{elem.id}_outliers")
                y = label_outls(outliers, elem.id, 10)
                if (not all(data_label == 0 for data_label in y) and
                        not all(data_label == 1 for data_label in y)):
                    outliers = remove_noise_features(outliers)
                    new_df = pd.concat(
                        [outliers_df[outl_method_name][0], outliers])
                    new_y = outliers_df[outl_method_name][1] + y
                    outliers_df[outl_method_name] = (new_df, new_y)
    
    # Mean Silhouette Score of the unsupervised methods of clustering
    mean_dbscan_score /= len(time_seriess_df_w_nf)
    mean_optics_score /= len(time_seriess_df_w_nf)
    mean_ocsvm_score  /= len(time_seriess_df_w_nf)
    # Printing the mean Silhouette Score of the unsupervised methods of clustering
    print("---------------------------Mean Silhouette Score------------------------------------")
    print("Mean Silhouette score DBSCAN: {}".format(mean_dbscan_score))
    print("Mean Silhouette score OPTICS: {}".format(mean_optics_score))
    print("Mean Silhouette score OCSVM: {}".format(mean_ocsvm_score))
    print("------------------------------------------------------------------------------------")

    for outl_method_name in outliers_df.keys():
        output_csv_idx = 1
        X, y = outliers_df[outl_method_name]
        if len(X):
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
                        "f1_score": result["mean_test_score"].values[:1][0]
                    }
                    save_to_json(
                    config, f"{best_configs_dir}/configs.json")
                    export_df_to_csv(
                    result, f"{model_name}-Results-{output_csv_idx}")

            output_csv_idx += 1


if __name__ == "__main__":
    main()
