import pandas as pd


def tsv_to_dataframe(file_path):
    dataframe = pd.read_csv(file_path, sep="\t")
    return dataframe


def csv_to_dataframe(file_path):
    dataframe = pd.read_csv(file_path, sep="\t")
    return dataframe
