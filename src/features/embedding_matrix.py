import os
import numpy as np
from gensim.models import KeyedVectors


def get_embedding_model_fasttext(file_path, num_vectors):
    emb_model = KeyedVectors.load_word2vec_format(
        file_path,
        limit=num_vectors,
    )
    return emb_model


def get_embedding_model_glove(file_path):
    f = open(os.path.join(file_path))
    emb_model = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        emb_model[word] = coefs
    f.close()
    return emb_model


def get_embedding_matrix(emb_model, X, emb_dim, num_words, word_index):
    """
    Creates an embedding matrix from pretrained emb_model.
    If word doesn't exist in emb_model the vector for that
    word will be the zero vector.
    word_index is from the keras_padded_w2i module.
    """
    embedding_matrix = np.zeros((num_words, emb_dim))
    num_oov = 0
    for word, i in word_index.items():
        if word in emb_model:
            embedding_vector = emb_model[word]
            embedding_matrix[i] = embedding_vector
        else:
            num_oov += 1
    return embedding_matrix, num_oov
