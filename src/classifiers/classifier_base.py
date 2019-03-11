import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from abc import ABCMeta, abstractmethod
import seaborn
from src.utils import logger


class Classifier(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        save_to_file=False,
        file_name=None
    ):
        pass

    @abstractmethod
    def predict(self, X, model=None):
        pass

    def confusion_matrix(self, y_true, y_pred, num_categories, names):
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        matrix_proportions = np.zeros((num_categories, num_categories))
        for i in range(0, num_categories):
            matrix_proportions[i, :] = \
                confusion_matrix[i, :] / float(confusion_matrix[i, :].sum())
        confusion_df = pd.DataFrame(
            matrix_proportions, index=names, columns=names)
        return confusion_df

    def f1_score(self, y_true, y_pred):
        return metrics.f1_score(y_true, y_pred, average=None)

    def recall(self, y_true, y_pred):
        return metrics.recall_score(y_true, y_pred, average=None)

    def precision(self, y_true, y_pred):
        return metrics.precision_score(y_true, y_pred, average=None)

    def accuracy(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    def plot_confusion_matrix(self, confusion_df, file_path=None):
        plt.rc('pdf', fonttype=42)
        plt.rcParams['ps.useafm'] = True
        plt.rcParams['pdf.use14corefonts'] = True
        plt.rcParams['text.usetex'] = True
        plt.figure(figsize=(5, 5))
        seaborn.heatmap(
            confusion_df,
            annot=True,
            annot_kws={"size": 12},
            cmap='gist_gray_r',
            cbar=False,
            square=True,
            fmt='.2f'
        )
        plt.ylabel(r'\textbf{True categories}', fontsize=14)
        plt.xlabel(r'\textbf{Predicted categories}', fontsize=14)
        plt.tick_params(labelsize=12)
        if file_path:
            plt.savefig(file_path)
        else:
            plt.show()

    def log_metrics(self, y_true, y_pred, y_map):
        recall = self.recall(y_true, y_pred)
        precision = self.precision(y_true, y_pred)
        f1_score = self.f1_score(y_true, y_pred)
        accuracy = self.precision(y_true, y_pred)
        self.logging.info("Index Map: {}".format(y_map))
        self.logging.info("Recall: {}".format(recall))
        self.logging.info("Precision: {}".format(precision))
        self.logging.info("F1 Score: {}".format(f1_score))
        self.logging.info("Accuracy: {}".format(accuracy))

    def true_vs_pred_to_csv(self, file_path, X_original, X_test, y_pred, y_true):
        # Make sure everything is of right datatype
        X_original = pd.DataFrame(X_original)
        X_test = pd.DataFrame(X_test)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        # Select out the rows from X_original which
        # are in X_test
        X_original = X_original[X_original.index.isin(X_test.index)]
        # Add columns to dataframe
        X_original["y_true"] = y_true.tolist()
        X_original["y_pred"] = y_pred.tolist()
        # Write to csv file
        mapping = {0: "NOT", 1: "OFF"}
        X_original = X_original.replace({"y_true": mapping, "y_pred": mapping})
        X_original.to_csv(file_path)

