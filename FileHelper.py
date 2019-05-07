import os
import pickle


def save_obj(obj, name):
    path_and_file = 'obj/' + name + '.pkl'
    path, _ = os.path.split(path_and_file)
    if not os.path.isdir(path):
        os.makedirs(path)

    with open(path_and_file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    path_and_file = 'obj/' + name + '.pkl'
    path, _ = os.path.split(path_and_file)
    if not os.path.exists(path) or not os.path.isfile(path_and_file):
        return None

    with open(path_and_file, 'rb') as f:
        return pickle.load(f)
