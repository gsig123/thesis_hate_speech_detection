import os
from datetime import datetime
from ..plotting.confusion_matrix import plot_confusion_matrix
from ..plotting.train_val_comparison import (
    plot_train_val_accuracy,
    plot_train_val_loss,
)
from ..utils.save_load_keras_model import save_model

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
    regularizations_line = "Regularization Amount: {}\n".format(
        regularizations)
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


def create_result_txt(
    dir_path,
    y_mapping,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix_df,
    train_size,
    val_size,
    test_size,
):
    # Num Train Samples
    # Num Val Samples
    # Num Test Samples
    file_name = dir_path + "/result.txt"
    y_mapping_line = "Y Mapping: {}\n".format(y_mapping)
    train_size_line = "Train Size: {}\n".format(train_size)
    test_size_line = "Test Size: {}\n".format(test_size)
    val_size_line = "Validation Size: {}\n".format(val_size)
    f1_line = "F1: {}\n".format(f1_score)
    recall_line = "Recall: {}\n".format(recall_score)
    precision_line = "Precision: {}\n".format(precision_score)
    confusion_line = "Confusion Matrix:\n{}\n".format(confusion_matrix_df)
    f = open(file_name, "w+")
    f.write(y_mapping_line)
    f.write(train_size_line)
    f.write(test_size_line)
    f.write(val_size_line)
    f.write(f1_line)
    f.write(recall_line)
    f.write(precision_line)
    f.write(confusion_line)
    f.close()
    return file_name


def train_val_loss_file_path(dir_path):
    return dir_path + "/train_val_loss.png"


def train_val_accuracy_file_path(dir_path):
    return dir_path + "/train_val_accuracy.png"


def confusion_matrix_file_path(dir_path):
    return dir_path + "/confusion_matrix.png"


def create_plots_save_model(dir_path, y_true, y_pred, y_map, history, model):
    save_model(model, history, dir_path)
    acc_file_path = train_val_accuracy_file_path(dir_path)
    loss_file_path = train_val_loss_file_path(dir_path)
    confusion_file_path = confusion_matrix_file_path(dir_path)
    plot_confusion_matrix(
        y_true, y_pred, len(y_map), list(y_map.keys()), confusion_file_path)
    plot_train_val_accuracy(history, acc_file_path)
    plot_train_val_loss(history, loss_file_path)
