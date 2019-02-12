import numpy as np 
import pandas as pd

file_path = "./data/OffensEval2019/start-kit/training-v1/offenseval-training-v1.tsv"

# Read in tsv file
df = pd.read_csv(file_path, sep='\t')

# Print headers (column values)
print(df.columns.values)
# Print first row
print(df.iloc[0])
# Map 'OFF' to 1 and 'NOT' to 0 in 'subtask_a column
df['subtask_a'] = df['subtask_a'].map({"OFF": 1, "NOT": 0})
print(df.iloc[[0, 1, 2, 3, 4, 5]])

# Get tweets as a numpy array
X = df['tweet'].values
print(X)
print(X.shape)
print(X.dtype)
y_task_a = df['subtask_a'].values
print(y_task_a)
