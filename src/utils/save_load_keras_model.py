from keras.models import model_from_yaml
from .pickle import save_to_pkl, read_from_pkl


def save_model(model, history, dir_path):
    model_yaml_path = dir_path + "/model.yaml"
    model_weights_path = dir_path + "/model_weights.h5"
    model_history_path = dir_path + "/model_history.pkl"
    model_yaml = model.to_yaml()
    with open(model_yaml_path, "w") as yaml_file:
        yaml_file.write(model_yaml)
    model.save_weights(model_weights_path)
    save_to_pkl(history, model_history_path)


def read_model(dir_path):
    model_yaml_path = dir_path + "/model.yaml"
    model_weights_path = dir_path + "/model_weights.h5"
    model_history_path = dir_path + "/model_history.pkl"
    yaml_file = open(model_yaml_path, "r")
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    loaded_model.load_weights(model_weights_path)
    model_history = read_from_pkl(model_history_path)
    return loaded_model, model_history
