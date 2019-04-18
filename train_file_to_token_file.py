import sys
from src.features.pre_trained_embedding_pipeline import (
    embedding_pipeline_tokens,
)
from src.preprocess.offens_eval import get_X_and_ys


def main(train_file_path, token_file_path):
    data = get_X_and_ys(train_file_path)
    X = data[0]
    tokens = embedding_pipeline_tokens(X)
    with open(token_file_path, "w+") as f:
        for token in tokens:
            f.write(token + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
