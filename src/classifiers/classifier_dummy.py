import numpy as np
from .classifier_base import Classifier


class DummyClassifier(Classifier):
    def __init__(self):
        self.dummy = "dummy"

    def prepare_dataset(self, dataframe):
        print("Prep dataset")

    def fit(self, X_train, y_train):
        print("Fit!")

    def predict(self, X):
        return np.zeros(X.shape[0])