import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from abc import ABCMeta, abstractmethod


class DataPrep(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    def tsv_to_dataframe(self, file_path):
        dataset = pd.read_csv(file_path, sep='\t')
        return dataset

    def csv_to_dataframe(self, file_path):
        dataset = pd.read_csv(file_path)
        return dataset

    def get_X_y(
            self,
            dataset,
            X_column_names=['tweet'],
            y_column_names=['subtask_a']):
        """
        y are categorical dependent variables, therefore
        they are tranformed to single boolean columns for each category.
        Returns X, y and a list of the new column names for y.
        """
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
        """
        Probably only usable for number features - not sure if this will be used.
        """
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        return X_train, X_test

    def handle_missing_data(self):
        """
        Added this method to remember we need to deal with 'NaN' values.
        """
        pass

    def transform_class_column_to_ints(self, dataframe, column_name, mapping):
        """
        Takes in a dataframe and changes a class column
        (where the classes are something else than ints) to
        a column with ints based on the mapping dictionary.
        Returns the transformed dataframe.
        """
        dataframe[column_name] = dataframe[column_name].map(mapping)
        return dataframe

    def remove_rows_where_column_is_NULL(self, dataframe, column_name):
        return dataframe[pd.notnull(dataframe[column_name])]
