from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn


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


def plot_confusion_matrix(
    y_true, y_pred, num_categories, names, save_to_file_path=None
):
    confusion_df = confusion_matrix(y_true, y_pred, num_categories, names)
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
        fmt='.2f',
    )
    plt.ylabel(r'\textbf{True categories}', fontsize=14)
    plt.xlabel(r'\textbf{Predicted categories}', fontsize=14)
    plt.tick_params(labelsize=12)
    if save_to_file_path:
        plt.savefig(save_to_file_path)
    else:
        plt.show()
