import sys
sys.path.append("..")
from classifiers.classifier_hsaofl import HSAOFLClassifier as Classifier
from preprocess.data_prep import DataPrep
from sklearn.metrics import classification_report

c = Classifier()
dp = DataPrep()
dataframe = dp.csv_to_dataframe(
    "../../data/raw/HateSpeechAndOffensiveLanguage/labeled_data.csv")
X, y, feature_names = c.prepare_dataset(dataframe)
print("X shape: {}, y shape: {}".format(X.shape, y.shape))
model, X_ = c.fit(X, y)
y_preds = c.predict(X_, model)
confusion_df = c.confusion_matrix(
    y_true=y, y_pred=y_preds, num_categories=3,
    names=["Hate", "Offensive", "Neither"])
f1_score = c.f1_score(y, y_preds)
recall = c.recall(y, y_preds)
precision = c.precision(y, y_preds)
accuracy = c.accuracy(y, y_preds)
print(confusion_df)
print("F1: {}, recall: {}, precision: {}, accuracy: {}".format(
    f1_score, recall, precision, accuracy))
report = classification_report(y, y_preds)
print(report)
c.plot_confusion_matrix(confusion_df=confusion_df)
