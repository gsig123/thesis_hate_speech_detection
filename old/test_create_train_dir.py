import os
from datetime import datetime

ROOT_PATH = "./model_training"


def create_train_dir(model_name):
    date_stamp = datetime.now().strftime("%Y-%m-%d")
    path = ROOT_PATH + "/" + model_name + "_" + date_stamp
    os.mkdir(path)
    return path