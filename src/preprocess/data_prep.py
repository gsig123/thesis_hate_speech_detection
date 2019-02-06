import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class DataPrep:

    def __init__(self):
        pass

    def file_to_dataframe(self, file_path):
        dataset = pd.read_csv(file_path, sep='\t')
        return dataset

    def get_X_y(self, dataset, X_column_names=['tweet'], y_column_names=['subtask_a']):
        y_df = pd.get_dummies(dataset[y_column_names])
        y_new_col_names = list(y_df)
        y = y_df.values
        X = dataset[X_column_names].values
        return X, y, y_new_col_names

    def train_test_split(self, X, y, test_size=0.2, random_state=0):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def feature_scaling(self, X_train, X_test):
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    def handle_missing_data(self):
        pass

    def categorical_data_to_boolean_columns(self, my_list):
        my_list = pd.get_dummies(df(my_list))
        return my_list

    