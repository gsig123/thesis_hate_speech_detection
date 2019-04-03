import pickle


def save_to_pkl(obj, file_path):
    """
    file_path should have a .pkl extension
    """
    f = open(file_path, "wb+")
    pickle.dump(obj, f)
    f.close()


def read_from_pkl(file_path):
    """
    file_path should have a .pkl extension
    """
    f = open(file_path, "rb+")
    obj = pickle.load(f)
    f.close()
    return obj