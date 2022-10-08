import pickle
from os import mkdir, path


def serialize_data(data, file_name: str) -> None:
    """
    Serialize object into a pickle file

    data: data to serialize
    file_name: name of the file to where store the serialized data
    """

    if not path.exists("./serialized_data"):
        mkdir("./serialized_data")

    make_pickle_file(file_name, data)


def make_pickle_file(file_name, data) -> None:
    """
    Create pickle file

    data: data to serialize
    file_name: name of the file to where store the serialized data
    """

    with open(f"{file_name}.pickle", "wb") as outfile:
        pickle.dump(data, outfile)


def deserialize_data(file_name) -> None:
    """
    Deserialize pickle file into an object

    file_name: name of the file to deserialize
    """

    if path.exists(file_name):
        data = unpick_pickle_file(file_name)
        return data


def unpick_pickle_file(file_name) -> None:
    """
    Get data from pickle file

    file_name: name of the file from where to deserialize the data
    """

    with open(file_name, "rb") as f:
        data = pickle.load(f)

    return data
