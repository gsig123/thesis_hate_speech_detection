import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from abc import ABCMeta, abstractmethod
import seaborn


class Classifier(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X_train, y_train, save_to_file=False, file_name=None):
        """
        - Trains and returns a model. 
        - If you set save_to_file to 'True' and provide a the file_name
          it will save the model as a '.pkl' file in the './models' directory
        - It will also create a textfile with some metadata about the model
          in the './models/meta' directory, with the same name as the model
        - Returns the trained model
        """
        pass

    @abstractmethod
    def predict(self, X, model=None):
        pass

    def confusion_matrix(self, y_true, y_pred, num_categories, names):
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        matrix_proportions = np.zeros((num_categories, num_categories))
        for i in range(0, 3):
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
        plt.show()
