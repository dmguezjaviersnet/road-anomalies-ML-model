from pathlib import Path
from os.path import exists

# //\\//  -------------------- Required directories for the data ----------------------- //\\\// #
data_dir = Path("./data")
csvs_dir = Path("./data/csvs")
marks_dir = Path("./data/marks")
samples_dir = Path("./data/samples")
proc_samples_dir = Path("./data/csvs/proc_samples") 
marks_google_dir = Path("./data/csvs/marks")

def create_req_dirs() -> None:
    """
    Create the required directories in case they doesn't exist

    """

    required_dirs = [data_dir, csvs_dir, marks_dir, samples_dir, proc_samples_dir, marks_google_dir]
    
    for req_dir in required_dirs:
        if not exists(req_dir):
            req_dir.mkdir(parents=True, exist_ok=True)
