import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from abc import ABCMeta, abstractmethod

class Classifier(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def prepare_dataset(self, dataframe):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    def confusion_matrix(self, y_true, y_pred):
        return metrics.confusion_matrix(y_true, y_pred)

    def f1_score(self, y_true, y_pred):
        return metrics.f1_score(y_true, y_pred)

    def recall(self, y_true, y_pred):
        return metrics.recall_score(y_true, y_pred)
    
    def precision(self, y_true, y_pred):
        return metrics.precision_score(y_true, y_pred)

    def accuracy(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    def plot_confusion_matrix(self, confusion_matrix, class_names, normalize=False, title="", cmap=plt.cm.Blues):
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        fmt = '.2f' if normalize else 'd'
        thresh = confusion_matrix.max() / 2.

        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                        horizontalalignment="center",
                         color="white" if confusion_matrix[i, j] > thresh else "black")

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        plt.show()