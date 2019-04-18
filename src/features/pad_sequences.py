from keras.preprocessing.sequence import pad_sequences as keras_pad_sequences


def pad_sequences(sequences, max_seq_len):
    padded_sequences = keras_pad_sequences(sequences, maxlen=max_seq_len)
    return padded_sequences
