from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def get_padded_w2i_matrix(X, max_num_words, max_seq_len):
    """
    Tokenizes the most frequent words (max_num_words).
    Considers sequences that are <= max_seq_len
    (max_seq_len num words in sentence).
    Creates word2index representation, padded so
    all sequences are of the same length so they are the same length.
    """
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    word_index = tokenizer.word_index  # Num of unique tokenss
    # Pad everything to the same length to be keras friendly
    X = pad_sequences(sequences, maxlen=max_seq_len)
    return X, word_index