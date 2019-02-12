from src.preprocess.data_prep import *
d = DataPrep()
dataset = d.tsv_to_dataframe("data/OffensEval2019/start-kit/training-v1/offenseval-training-v1.tsv")
X, y, y_col_names = d.get_X_y(dataset, y_column_names=['subtask_a', 'subtask_b', 'subtask_c'])
print(X)
print(y)
print(y_col_names)
