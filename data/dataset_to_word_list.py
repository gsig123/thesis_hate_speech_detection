import pandas as pd
import numpy as np
import nltk
import string


def dataset_to_word_list(input_file_path):
    df = pd.read_csv(input_file_path, sep="\t")
    tweets = df["tweet"].values
    unique_tokens = []
    for tweet in tweets:
        tokens = nltk.word_tokenize(tweet)
        for token in tokens:
            token = token.lower()  # Lower case
            token = token.translate(str.maketrans(
                '', '', string.punctuation))  # Remove punctuation
            unique_tokens.append(token)
    unique_tokens = list(set(unique_tokens))
    return unique_tokens


input_file_path = "./raw/OffensEval2019/start-kit/training-v1/offenseval-training-v1.tsv"
output_file_path = "./processed/OffensEval2019/english_word_list.txt"


def write_word_list_to_file(unique_word_list, output_file_path):
    with open(output_file_path, "w") as f:
        for word in unique_word_list:
            f.write(word + "\n")


unique_word_list = dataset_to_word_list(input_file_path)
write_word_list_to_file(unique_word_list, output_file_path)
