from pyfasttext import FastText
import numpy as np
from .tokenize import tokenize


def get_fasttext_vectors(X):
    model = FastText('./models/fasttext/OE2019-training-v1-100d-model.bin')
    tweet_vectors = []
    for tweet in X:
        tweet_vector = []
        words = tokenize(tweet)
        for word in words:
            tweet_vector.append(np.array(model[word]))
        tweet_vectors.append(np.array(tweet_vector))
    return np.array(tweet_vectors)


def create_fasttext_emb_matrix(X):
    model = FastText('./models/fasttext/OE2019-training-v1-100d-model.bin')
    emb_matrix = []
    seen_words = {}
    emb_matrix.append(np.zeros(100))  # Add 0 index (padding)
    for tweet in X:
        words = tokenize(tweet)
        for word in words:
            if word not in seen_words:
                emb = model[word]
                emb_matrix.append(np.array(emb))
                seen_words[word] = True
    return np.array(emb_matrix)
