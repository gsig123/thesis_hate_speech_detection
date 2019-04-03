from .tokenize import tokenize
from tensorflow import keras
import numpy as np


def w2i(list_of_sentences, padded=True):
    """
    Input:
    - List of sentences (strings)
    - A boolean varable, 'padded', which indicates whether or
      not the output should be padded so each list has the
      same length.
    Output:
    - X: A list of lists, where each inner list contains an
      integer representing a word.
    - w2i: A word to index dictionary. A padding index is located available
           with: w2i["<PAD">] = 0
    - i2w: A index to word dictionary (the inverse of w2i).
    """
    X = []
    w2i = {}
    w2i_curr_index = 0
    max_word_count = 0
    w2i["<PAD>"] = 0
    w2i_curr_index += 1
    for sentence in list_of_sentences:
        w2i_sentence = []
        words = tokenize(sentence)
        for word in words:
            if word not in w2i:
                w2i[word] = w2i_curr_index
                w2i_curr_index += 1
            w2i_sentence.append(w2i[word])
        X.append(np.array(w2i_sentence))
        if len(w2i_sentence) > max_word_count:
            max_word_count = len(w2i_sentence)
    X = np.array(X)
    if padded:
        X = keras.preprocessing.sequence.pad_sequences(
            X,
            value=w2i["<PAD>"],
            padding="post",
            maxlen=max_word_count,
        )
    i2w = dict([(value, key) for (key, value) in w2i.items()])
    return X, w2i, i2w
