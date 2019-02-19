from .classifier_base import Classifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


class HSAOFLClassifier(Classifier):
    def __init__(self,
                 class_weight_1,
                 class_weight_2,
                 class_weight_3,
                 penalty_1,
                 penalty_2,
                 penalty_3,
                 c_1,
                 c_2,
                 c_3,
                 loss_function,
                 multi_class):
        self.class_weight_1 = class_weight_1
        self.class_weight_2 = class_weight_2
        self.class_weight_3 = class_weight_3
        self.penalty_1 = penalty_1
        self.penalty_2 = penalty_2
        self.penalty_3 = penalty_3
        self.c_1 = c_1
        self.c_2 = c_2
        self.c_3 = c_3
        self.loss_function = loss_function
        self.multi_class = multi_class

    def fit(self, X, y):
        select = SelectFromModel(
            LogisticRegression(
                class_weight=self.class_weight_1,
                penalty=self.penalty_1,
                C=self.c_1,
            )
        )
        X_ = select.fit_transform(X, y)
        model = LinearSVC(
            class_weight=self.class_weight_2,
            C=self.c_2,
            penalty=self.penalty_2,
            loss=self.loss_function,
            multi_class=self.multi_class,
        ).fit(X_, y)
        model = LogisticRegression(
            class_weight=self.class_weight_3,
            penalty=self.penalty_3,
            C=self.c_3,
            solver='lbfgs',
        ).fit(X_, y)
        return model, X_

    def predict(self, X, model):
        y_preds = model.predict(X)
        return y_preds
