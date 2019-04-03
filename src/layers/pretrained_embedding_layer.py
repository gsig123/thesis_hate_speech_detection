from keras.layers import Embedding


def get_pretrained_embedding_layer(
    num_unique_words,
    emb_dim,
    emb_matrix,
    max_sequence_len,
):
    embedding_layer = Embedding(
        num_unique_words,
        emb_dim,
        weights=[emb_matrix],
        input_length=max_sequence_len,
        trainable=False,
    )
    return embedding_layer
