from keras.preprocessing.text import Tokenizer


def keras_tokenize(X, num_words):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    w2i = tokenizer.word_index
    return sequences, w2i
