import os
from datetime import datetime

ROOT_PATH = "./model_training"


def create_train_dir(model_name):
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = ROOT_PATH + "/" + model_name + "_" + time_stamp
    os.mkdir(path)
    return path


def create_meta_txt(
    dir_path,
    model_name,
    train_path,
    model,
    units,  # List
    dropouts,  # List
    regularizations,  # List
    activation_functions,  # List
    optimizer,
    loss,
    metric,
    epochs,
    batch_size,
):
    file_name = dir_path + "/meta.txt"

    model_name_line = "Model Name: {}\n".format(model_name)
    training_file_line = "Training File: {}\n".format(train_path)
    dropout_line = "Dropout Amount: {}\n".format(dropouts)
    regularizations_line = "Regularization Amount: {}\n".format(regularizations)
    units_line = "Units: {}\n".format(units)
    activation_line = "Activation Functions: {}\n".format(activation_functions)
    optimzer_line = "Optimizer: {}\n".format(optimizer)
    loss_line = "Loss Function: {}\n".format(loss)
    metrics_line = "Metric: {}\n".format(metric)
    epochs_line = "Epochs: {}\n".format(epochs)
    batch_size = "Batch Size: {}\n".format(batch_size)

    f = open(file_name, "w+")
    f.write(model_name_line)
    f.write(training_file_line)
    f.write(dropout_line)
    f.write(regularizations_line)
    f.write(units_line)
    f.write(activation_line)
    f.write(optimzer_line)
    f.write(loss_line)
    f.write(metrics_line)
    f.write(epochs_line)
    f.write(batch_size)
    f.write(dropout_line)
    f.write("Model Details:\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.close()
    return file_name


def create_result_text(
    dir_path,
    y_mapping,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix_df,
):
    # Num Train Samples
    # Num Val Samples
    # Num Test Samples
    file_name = dir_path + "/meta.txt"
    y_mapping_line = "Y Mapping: {}\n".format(y_mapping)
    f1_line = "F1: {}\n".format(f1_score)
    recall_line = "Recall: {}\n".format(recall_score)
    precision_line = "Precision: {}\n".format(precision_score)
    confusion_line = "Confusion Matrix: {}\n".format(confusion_matrix_df)
    f = open(file_name, "w+")
    f.write(y_mapping_line)
    f.write(f1_line)
    f.write(recall_line)
    f.write(precision_line)
    f.write(confusion_line)
    f.close()


def train_val_loss_file_path(dir_path):
    return dir_path + "/train_val_loss.png"


def train_val_accuracy_file_path(dir_path):
    return dir_path + "/train_val_accuracy.png"


def confusion_matrix_file_path(dir_path):
    return 