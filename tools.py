from os import mkdir, path
import pickle

def make_pickle_file(file_name, data) -> None:
    """
    Create pickle file
    """

    with open(f"{file_name}.pickle", "wb") as outfile:
        pickle.dump(data, outfile)


def unpick_pickle_file(file_name) -> None:
    """
    Get data from pickle file
    """

    with open(file_name, "rb") as f:
        data = pickle.load(f)

    return data


def serialize_data(data, file_name: str) -> None:
    """
    Serialize object into a pickle file
    """

    if not path.exists("./serialized_data"):
        mkdir("./serialized_data")

    make_pickle_file(file_name, data)


def deserialize_data(file_name) -> None:
    """
    Deserialize pickle file into an object
    """

    if path.exists(file_name):
        data = unpick_pickle_file(file_name)
        return data

    else:
        return None
