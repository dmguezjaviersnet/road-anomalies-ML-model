from unittest import result
from tabulate import tabulate
from IPython.display import display

from sys import argv
import time
from data_processing import export_df_to_csv, marks_json_to_df, json_samples_to_df
from features_processing import add_features, feature_selection, remove_noise_features, load_feature_df_count, update_csv_feature_df_count
from outls_labeling import label_outls
from outls_plots import outls_scatter
from outls_detection import detect_outls, filter_outliers
from windowing_process import build_windows
from tools import samples_dir, marks_dir, create_req_dirs, save_to_json, best_configs_dir, serialized_preproce_data_dir
from model_selection import select_model, export_confusion_matrix, buil_train_test_set, export_locations_outputs_df_to_csv, keep_only_featured_selected
import pandas as pd
from os.path import exists
from serializer import serialize_data, deserialize_data
from tools import generate_unique_id_matrix, images_dir, valid_model_names, output_rtest_locations_dir
import copy

def main(args):
    model_input_name: str = "knn"
    recompute_data: bool =  False

    if len(args) > 1:
        if args[1] in valid_model_names:
            model_input_name = args[1]
        else:
            raise Exception("Invalid model name")
    if len(args) > 2:
        if args[2] == 'true' or args[2] == 'false':
            recompute_data = True if args[2] == 'true' else False
        else:
            raise Exception("Expected boolean value (true or false)")
    
    start_time = time.time()
    create_req_dirs()
    outliers_df: dict[str, tuple[pd.DataFrame, list]] = {}
    mean_dbscan_score = 0
    mean_optics_score = 0
    mean_ocsvm_score = 0
    series_count = 0
    serialized_data_filename: str = f"{serialized_preproce_data_dir}/preproce_data"
    print(serialized_data_filename)

    if exists(f"{serialized_data_filename}.pickle"):
        data: dict = deserialize_data(f"{serialized_data_filename}.pickle")
        outliers_df = data['outliers_df']
        mean_dbscan_score = data['mean_dbscan_score']
        mean_optics_score = data['mean_optics_score']
        mean_ocsvm_score = data['mean_ocsvm_score']
        series_count = data['series_count']
    else:
        marks_json_to_df(f"{marks_dir}")
        time_seriess_df = json_samples_to_df(f"{samples_dir}")

        time_seriess_df_w_nf = [add_features(elem) for elem in time_seriess_df]
        #export_df_to_csv(time_seriess_df_w_nf[0].series, "temporal")
        # //\\// ------------------ Taking outliers along the whole time series --------------------------//\\//
        outliers_df = {
            "z_thresh": (pd.DataFrame(), []),
            "z_diff": (pd.DataFrame(), []),
            "g_zero": (pd.DataFrame(), []),
            "dbscan": (pd.DataFrame(), []),
            "optics": (pd.DataFrame(), []),
            "ocsvm": (pd.DataFrame(), []),
        }
        series_count = len(time_seriess_df_w_nf)
        for elem in time_seriess_df_w_nf:
            predictions, score_dbscan, score_optics, score_ocsvm = detect_outls(
                elem.series, elem.id)
            mean_dbscan_score += score_dbscan
            mean_ocsvm_score += score_ocsvm
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
        mean_dbscan_score /= series_count
        mean_optics_score /= series_count
        mean_ocsvm_score  /= series_count

    serialize_data({
        "outliers_df": outliers_df,
        "mean_dbscan_score": mean_dbscan_score,
        "mean_optics_score": mean_optics_score,
        "mean_ocsvm_score": mean_ocsvm_score,
        "series_count": series_count
    }, serialized_data_filename)

    # Printing the mean Silhouette Score of the unsupervised methods of clustering
    print("---------------------------Mean Silhouette Score------------------------------------")
    print("Mean Silhouette score DBSCAN: {}".format(mean_dbscan_score))
    print("Mean Silhouette score OCSVM: {}".format(mean_ocsvm_score))
    print("Mean Silhouette score OPTICS: {}".format(mean_optics_score))
    print("------------------------------------------------------------------------------------")
    
    #compute train and  test set

    

    #outliers_df.pop("g_zero")
    outliers_df.pop("optics")
    best_model_config = {}
    for outl_method_name in outliers_df.keys():
        output_csv_idx = 1
        X, y = outliers_df[outl_method_name]
        
                
        # if there is any outliers detected with outl_method_name
        print(f'Outlieres detected by {outl_method_name}: {len(X)}')
        if len(X) > 10:
            train_test_set: dict = {}
            train_test_filename = f"{serialized_preproce_data_dir}/{outl_method_name}_train_test_set"
            if exists(f"{train_test_filename}.pickle") and not recompute_data:
                train_test_set = deserialize_data(f"{train_test_filename}.pickle")
            else:
                train_test_set = buil_train_test_set(X, y)
                serialize_data(train_test_set, f"{train_test_filename}", recompute_data)
            # print(train_test_set["X_test_location"])
            best_outl_model = {}
            # distinct number of features to select for each method of feature selection
            X = X.drop(['Latitude', 'Longitude'], axis=1)
            for n_features_to_select in [3, 6, 9]:
                features_selected_sets = feature_selection(
                    X, y, n_features_to_select)

                print(
                    f"---------------- Features selected with outliers detected using {outl_method_name} ----------------\n")

                for selector_name, features_selected_set in features_selected_sets:
                    f_df = load_feature_df_count()
                    for feature in features_selected_set:
                        f_df[feature] = [f_df[feature].values[0] + 1]
                    update_csv_feature_df_count(f_df)
                    print(f"Features selected with selector {selector_name}")
                    print(f"{features_selected_set}\n")
                    # df_sel_feats: pd.DataFrame = pd.DataFrame(
                    #     X[features_selected_set])
                    new_train_test_set = copy.deepcopy(train_test_set)
                    new_train_test_set = keep_only_featured_selected(new_train_test_set, features_selected_set)
                    ms_results = select_model(new_train_test_set, model_input_name, True)
                    #confusion_m
                    for model_selected_name, result, prec, recall,  acc, f1, confusion_m , loc_points in ms_results:
                        print(f"----- Results with {model_selected_name} model and {outl_method_name} outlier mehotd-----")
                        print(f"{result}\n\n")
                        matrix_id = generate_unique_id_matrix(model_selected_name)

                        config = {
                            "outliers_method": outl_method_name,
                            "feature_selector": selector_name,
                            "features_selected_set": features_selected_set,
                            "model": model_selected_name,
                            "best-config": result["params"].values[:1][0],
                            "f1_score_train": result["mean_test_score"].values[:1][0],
                            "f1_score_test": f1,
                            "precision_test": prec,
                            "recall_test": recall,
                            "accuracy_test": acc,
                            "confusion_matrix_path": f"{images_dir}/{matrix_id}.png",
                            "location_points_path": f"{output_rtest_locations_dir}/{matrix_id}.csv",
                            "n_outliers": len(X)         
                            }
                        if best_model_config:
                            best_model_config =  config if f1 > best_model_config["f1_score_test"] else best_model_config
                        else:
                            best_model_config = config
                        if best_outl_model:
                            best_outl_model = config if f1 > best_outl_model["f1_score_test"] else best_outl_model
                        else:
                            best_outl_model = config
                        export_confusion_matrix(confusion_m, matrix_id)
                        
                        save_to_json(
                            config, f"{best_configs_dir}/{model_selected_name}-results.json", False)
                        export_df_to_csv(
                            result, f"{model_selected_name}-Results-{output_csv_idx}")
                        export_locations_outputs_df_to_csv(loc_points,f"{matrix_id}")
            output_csv_idx += 1
            model_name_best_config = best_outl_model["model"]
            save_to_json(best_outl_model, f"{best_configs_dir}/{model_name_best_config}-results.json", True, f"best-{outl_method_name}")


    best_model_name = best_model_config["model"]
    save_to_json(best_model_config, f"{best_configs_dir}/{best_model_name}-results.json", True, "best")
    
    print(f"--------Time to complete code {(time.time() - start_time)//60} minutes and {round((time.time() - start_time)%60, 2)} seconds ----" )        

if __name__ == "__main__":
    main(argv)
