import json

def read_json(filename: str):
    f = open(filename)
    data = json.load(f)
    f.close()
     
    return data

def get_data(label: str, file: str):
    json_data = read_json(file)
    features_amount = len(json_data[0][label])
    features = [[]]*features_amount

    for elem in json_data:
        data = elem[label]
        for i in range(features_amount):
            features[i].append(data[i])

    return features
