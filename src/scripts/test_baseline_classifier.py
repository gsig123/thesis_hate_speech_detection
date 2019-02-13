import sys
sys.path.append("..")
from classifiers.classifier_baseline import BaselineClassifier as Classifier
from preprocess.data_prep_baseline import DataPrepBaseline
from sklearn.metrics import classification_report

c = Classifier(
    class_weight_1='balanced',
    class_weight_2='balanced',
    class_weight_3='balanced',
    penalty_1='l1',
    penalty_2='l2',
    penalty_3='l2',
    c_1=0.01,
    c_2=0.01,
    c_3=0.01,
    loss_function='squared_hinge',
    multi_class='ovr',
)
dp = DataPrepBaseline(language="english")
dataframe = dp.tsv_to_dataframe(
    "../../data/raw/OffensEval2019/start-kit/training-v1/offenseval-training-v1.tsv")
dataframe = dp.transform_class_column_to_ints(dataframe,
                                              column_name="subtask_a",
                                              mapping={"OFF": 1, "NOT": 0})
X, y, feature_names = dp.get_X_y_feature_names(dataframe, "tweet", "subtask_a")
print("X shape: {}, y shape: {}".format(X.shape, y.shape))
model, X_ = c.fit(X, y)
y_preds = c.predict(X_, model)
confusion_df = c.confusion_matrix(
    y_true=y,
    y_pred=y_preds,
    num_categories=2,
    names=["Not", "OFF"],
)
f1_score = c.f1_score(y, y_preds)
recall = c.recall(y, y_preds)
precision = c.precision(y, y_preds)
accuracy = c.accuracy(y, y_preds)
print(confusion_df)
print("F1: {}, recall: {}, precision: {}, accuracy: {}".format(
    f1_score, recall, precision,   accuracy))
report = classification_report(y, y_preds)
print(report)
c.plot_confusion_matrix(confusion_df=confusion_df)
