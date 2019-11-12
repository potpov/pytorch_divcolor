import pickle
import time
import json


def save_dict_to_pickle(d, pickle_path):
    # Store data (serialize)
    with open(pickle_path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


# load dict from pickle
def load_dict_from_pickle(pickle_path):
    # Load data (deserialize)
    with open(pickle_path, 'rb') as handle:
        unserialized_data = pickle.load(handle)
        return unserialized_data


# save the dict as a json
def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=4)


# load dict from json
def load_dict_from_json(json_path):
    with open(json_path) as f:
        params = json.load(f)
        return params


if __name__ == '__main__':
    # create a simple dict
    data = {'foo': 'bar'}
    save_dict_to_pickle(data, 'filename.pickle')
    unserialized_data = load_dict_from_pickle('filename.pickle')
    # check the data

    save_dict_to_json(data, 'filename.json')
    dict_from_json = load_dict_from_json('filename.json')

    print(data == unserialized_data)
    print(data == dict_from_json)
