import pandas as pd
import numpy as np


def dataset_to_txt(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path, sep="\t")
    tweets = df['tweet'].values
    np.savetxt(output_file_path, tweets, fmt="%s")


input_file_path = "./raw/OffensEval2019_Danish/danish_1600.tsv"
output_file_path = "./processed/OffensEval2019_Danish/danish_1600.txt"

dataset_to_txt(input_file_path, output_file_path)
