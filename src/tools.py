import json
import os
from pathlib import Path
from os.path import exists
import pandas as pd

# //\\//  -------------------- Required directories for the data ----------------------- //\\\// #
data_dir = Path("./data")
csvs_dir = Path("./data/csvs")
marks_dir = Path("./data/marks")
samples_dir = Path("./data/samples")
proc_samples_dir = Path("./data/csvs/proc_samples") 
marks_google_dir = Path("./data/csvs/marks")
test_csvs_dir = Path("./data/grid-search-results")
best_configs_dir = Path("./data/best-configs")

def create_req_dirs() -> None:
    '''
        Create the required directories in case they doesn't exist

    '''


    required_dirs = [data_dir, csvs_dir, marks_dir, samples_dir, proc_samples_dir, marks_google_dir, test_csvs_dir, best_configs_dir]
    
    for req_dir in required_dirs:
        if not exists(req_dir):
            req_dir.mkdir(parents=True, exist_ok=True)


def remove_parenthesis(s: str) -> str:
    '''
        Remove parenthesis from a string.

        Parameters
        ----------------

        s: String to remove parenthesis from.

        Returns
        ----------------

        String without parenthesis.

    '''

    return s.replace("(", "").replace(")", "")

def str_to_tuple(s: str) -> tuple:
    '''
        Convert a string to a tuple.

        Parameters
        ----------------

        s: String to convert to tuple.

        Returns
        ----------------

        Tuple with the values of the string.

    '''
    s = remove_parenthesis(s)
    return tuple(map(float, s.split(",")))

def remove_split_scores(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Remove the split scores from the dataframe.

        Parameters
        ----------------

        dt: Dataframe to remove the split scores from.

        Returns
        ----------------

        Dataframe without the split scores.

    '''

    regex = r"split\d+_test_score"
    return df[df.columns.drop(list(df.filter(regex=regex)))]


def save_to_json(new_data: dict, filename: str) -> None:
    '''
        Save a dataframe to a json file.

        Parameters
        ----------------

        df: Dataframe to save.
        filename: Name of the json file to save the dataframe to.

    '''
    if os.path.exists(filename):
        with open(filename,'r+') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
            # Join new_data with file_data inside configs
            file_data["configs"].append(new_data)
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file, indent = 4)
    else:
        with open(filename, 'w') as file:
            json.dump({"configs": [new_data]}, file, indent = 4)




    
