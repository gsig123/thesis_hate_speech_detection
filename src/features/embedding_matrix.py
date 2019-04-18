import os
import numpy as np
from gensim.models import KeyedVectors
from pyfasttext import FastText


def create_embedding_model_fasttext(file_path, num_vectors=None):
    if num_vectors:
        emb_model = KeyedVectors.load_word2vec_format(
            file_path,
            limit=num_vectors,
        )
    else:
        emb_model = KeyedVectors.load_word2vec_format(
            file_path,
        )
    return emb_model


def create_embedding_model_glove(file_path):
    f = open(os.path.join(file_path))
    emb_model = {}
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype="float32")
        except Exception:
            continue
        emb_model[word] = coefs
    f.close()
    return emb_model


def create_embedding_model(file_path):
    f = open(os.path.join(file_path))
    emb_model = {}
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype="float32")
        except Exception:
            continue
        emb_model[word] = coefs
    f.close()
    return emb_model


def create_embedding_matrix(emb_model, emb_dim, word_index, oov_model_path=None):
    """
    Creates an embedding matrix from pretrained emb_model.
    If word doesn't exist in emb_model the vector for that
    word will be the zero vector.
    word_index is from the keras_padded_w2i module.
    """
    num_words = len(word_index) + 1
    if oov_model_path:
        oov_model = FastText(oov_model_path)
    else:
        oov_model = None
    embedding_matrix = np.zeros((num_words, emb_dim))
    num_oov = 0
    for word, i in word_index.items():
        if word in emb_model and len(emb_model[word]) == emb_dim:
            embedding_vector = emb_model[word]
            embedding_matrix[i] = embedding_vector
        else:
            if oov_model:
                embedding_matrix[i] = oov_model[word]
            num_oov += 1
    return embedding_matrix, num_oov
