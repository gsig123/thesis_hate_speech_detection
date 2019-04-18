from .tokenize import tokenize
from .keras_tokenize import keras_tokenize
from .pad_sequences import pad_sequences
from .embedding_matrix import create_embedding_model, create_embedding_matrix
from ..CONSTANTS import MAX_SEQ_LEN, MAX_NUM_WORDS


def pre_trained_embedding_pipeline(
    X,
    embedding_file_path,
    max_seq_len=MAX_SEQ_LEN,
    num_words=MAX_NUM_WORDS,
    emb_dim=300,
    language="english",
    stem=False,
):
    # Step 1: Tokenize the sequences
    tokenized = []
    for sentence in X:
        tokenized.append(tokenize(sentence, language=language, stem=stem))
    # Step 2: Keras tokenization and sequence magic
    sequences, w2i = keras_tokenize(tokenized, num_words=num_words)
    # Step 3: Pad all sequences to max_seq_len
    padded_sequences = pad_sequences(sequences, max_seq_len=max_seq_len)
    # Step 4: Create the embedding model from pre-trained file
    emb_model = create_embedding_model(embedding_file_path)
    # Step 5: Create the embedding matrix
    emb_matrix, num_oov = create_embedding_matrix(
        emb_model,
        emb_dim,
        w2i,
    )
    return emb_matrix, num_oov, w2i, padded_sequences


def embedding_pipeline_tokens(
    X,
    max_seq_len=MAX_SEQ_LEN,
    num_words=MAX_NUM_WORDS,
    emb_dim=300,
    language="english",
    stem=False,
):
    """
    Takes in a train X and returns the same tokens as the
    'pre_trained_embedding_pipeline' returns.
    """
    # Step 1: Tokenize the sequences
    tokenized = []
    for sentence in X:
        tokenized.append(tokenize(sentence, language=language, stem=stem))
    # Step 2: Keras tokenization and sequence magic
    sequences, w2i = keras_tokenize(tokenized, num_words=num_words)
    return list(w2i)
