from sklearn import metrics


def f1_score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average=None)


def recall(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred, average=None)


def precision(y_true, y_pred):
    return metrics.precision_score(y_true, y_pred, average=None)


def accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)


def confusion_matrix(y_true, y_pred, num_categories, names):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    matrix_proportions = np.zeros((num_categories, num_categories))
    for i in range(0, num_categories):
        matrix_proportions[i, :] = \
            confusion_matrix[i, :] / float(confusion_matrix[i, :].sum())
    confusion_df = pd.DataFrame(
        matrix_proportions,
        index=names,
        columns=names,
    )
    return confusion_df
