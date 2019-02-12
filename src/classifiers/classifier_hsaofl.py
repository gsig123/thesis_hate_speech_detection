from .classifier_base import Classifier
from preprocess import data_prep_hsaofl
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


class HSAOFLClassifier(Classifier):
    def __init__(self):
        pass

    def prepare_dataset(self, dataframe):
        dp = data_prep_hsaofl.DataPrepHSAOFL()
        X, y, feature_names = dp.get_X_y_feature_names(dataframe)
        return X, y, feature_names

    def fit(self, X, y):
        select = SelectFromModel(LogisticRegression(
            class_weight='balanced', penalty="l1", C=0.01))
        X_ = select.fit_transform(X, y)
        model = LinearSVC(
            class_weight='balanced',
            C=0.01, penalty='l2',
            loss='squared_hinge',
            multi_class='ovr').fit(X_, y)
        model = LogisticRegression(
            class_weight='balanced', penalty='l2', C=0.01).fit(X_, y)
        return model, X_

    def predict(self, X, model):
        y_preds = model.predict(X)
        return y_preds
