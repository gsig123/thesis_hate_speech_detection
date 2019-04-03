from src.preprocess.data_prep_offenseval import DataPrepOffensEval
from src.classifiers.classifier_bi_lstm import BiLstmClassifier
from src.feature_extraction.w2i import w2i
import pandas as pd
import numpy as np


train_path = "./data/processed/OffensEval2019/start-kit/training-v1/train.csv"
test_path = "./data/processed/OffensEval2019/start-kit/training-v1/test.csv"
val_path = "./data/processed/OffensEval2019/start-kit/training-v1/val.csv"


dp = DataPrepOffensEval()

result_tuple_train = dp.get_X_and_ys(train_path)
result_tuple_test = dp.get_X_and_ys(test_path)
result_tuple_val = dp.get_X_and_ys(val_path)

sub_a_mapping = result_tuple_train[4]

X_train = result_tuple_train[0]
y_sub_a_train = result_tuple_train[1]
X_test = result_tuple_test[0]
y_sub_a_test = result_tuple_test[1]
X_val = result_tuple_val[0]
y_sub_a_val = result_tuple_val[1]

print("OFF Train Set: {}".format(np.nonzero(y_sub_a_train)[0].size / y_sub_a_train.size))
print("OFF  Val  Set: {}".format(np.nonzero(y_sub_a_val)[0].size / y_sub_a_val.size))
print("OFF Test  Set: {}".format(np.nonzero(y_sub_a_test)[0].size / y_sub_a_test.size))

X_train_w2i, train_w2i_dict, train_i2w_dict = w2i(X_train)
X_test_w2i, test_w2i_dict, test_i2w_dict = w2i(X_test)
X_val_w2i, val_w2i_dict, val_i2w_dict = w2i(X_val)

w2i_dict = {**train_w2i_dict, **test_w2i_dict, **val_w2i_dict}

classifier = BiLstmClassifier(
    embedding_input_dim=len(w2i_dict),
    epochs=10,
    batch_size=512,
    lstm_layers=50,
    mlp_layers=16,
    logfile="logloglog.txt",
)
model = classifier.fit(X_train_w2i, y_sub_a_train, X_val_w2i, y_sub_a_val)
classifier.plot_train_val_loss(file_path="loss.png")
classifier.plot_train_val_metric(file_path="metric.png")
y_test_pred = classifier.predict(X_test_w2i, model)
classifier.log_metrics(y_sub_a_test, y_test_pred, sub_a_mapping)
class_names = list(sub_a_mapping.keys())
confusion_df = classifier.confusion_matrix(y_sub_a_test, y_test_pred, 2, class_names)
classifier.plot_confusion_matrix(confusion_df, file_path="confusion.png")

