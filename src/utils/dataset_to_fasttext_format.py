import pandas as pd
import numpy as np


def dataset_to_fasttext_format(input_file, output_file):
    dataframe = pd.read_csv(input_file, sep="\t")
    tweets = dataframe["tweet"].values
    np.savetxt(output_file, tweets, fmt="%s")
