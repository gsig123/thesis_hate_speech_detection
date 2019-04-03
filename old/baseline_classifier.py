from src.classifiers.classifier_baseline import BaselineClassifier
from src.preprocess.data_prep_baseline import DataPrepBaseline
from src.preprocess.data_prep_offenseval import DataPrepOffensEval

c = BaselineClassifier(
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

dp = DataPrepBaseline(language='english')
dataframe = dp.tsv_to_dataframe(
    './data/raw/OffensEval2019/start-kit/training-v1/offenseval-training-v1.tsv')
dataframe = dp.transform_class_column_to_ints(
    dataframe=dataframe,
    column_name='subtask_a',
    mapping={'OFF': 1, 'NOT': 0},
)
X, y, feature_names = dp.get_X_y_feature_names(
    dataset=dataframe,
    tweet_column_name='tweet',
    y_column_name='subtask_a',
)
X_train, X_test, y_sub_a_train, y_sub_a_test = dp.train_test_split(
    X,
    y,
    test_size=0.2,
)
X_train, X_val, y_sub_a_train, y_sub_a_val = dp.train_test_split(
    X_train,
    y_sub_a_train,
    test_size=0.1,
)
model, X_, X_test = c.fit(X_train, y_sub_a_train, X_test)
y_pred = c.predict(X_test, model)

confusion_df_sub_a = c.confusion_matrix(
    y_true=y_sub_a_test,
    y_pred=y_pred,
    num_categories=2,
    names=['NOT', 'OFF'],
)
f1_score_sub_a = c.f1_score(y_sub_a_test, y_pred)
recall_sub_a = c.recall(y_sub_a_test, y_pred)
precision_sub_a = c.precision(y_sub_a_test, y_pred)
accuracy_sub_a = c.accuracy(y_sub_a_test, y_pred)
print("F1: {}, Recall: {}, Precision: {}, Accuracy: {}".format(
    f1_score_sub_a,
    recall_sub_a,
    precision_sub_a,
    accuracy_sub_a,
))

c.plot_confusion_matrix(confusion_df_sub_a, "confusion_matrix_baseline.png")
