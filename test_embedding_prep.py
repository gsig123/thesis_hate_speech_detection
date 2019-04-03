from src.preprocess.offens_eval import get_X_and_ys
from src.features.keras_padded_w2i import get_padded_w2i_matrix
from src.features.embedding_matrix import (
    get_embedding_model_fasttext,
    get_embedding_model_glove,
    get_embedding_matrix,
)
from src.layers.pretrained_embedding_layer import (
    get_pretrained_embedding_layer,
)
from src.CONSTANTS import (
    GLOVE_DIM,
    GLOVE_EN_PATH,
    FAST_TEXT_DIM,
    FAST_TEXT_EN_PATH,
    NUM_VECTORS,
    MAX_NUM_WORDS,
    MAX_SEQ_LEN,
    EN_FILE_PATH,
)

# Get the data...
data = get_X_and_ys(EN_FILE_PATH)
X = data[0]

# Create padded w2i matrix from X...
X, word_index = get_padded_w2i_matrix(X, MAX_NUM_WORDS, MAX_SEQ_LEN)

# Create the embedding matrices...
emb_model_glove = get_embedding_model_glove(GLOVE_EN_PATH)
emb_model_fasttext = get_embedding_model_fasttext(
    FAST_TEXT_EN_PATH, NUM_VECTORS)

emb_matrix_glove = get_embedding_matrix(
    emb_model_glove, X, GLOVE_DIM, len(word_index) + 1, word_index)
emb_matrix_fast_text = get_embedding_matrix(
    emb_model_fasttext, X, FAST_TEXT_DIM, len(word_index) + 1, word_index)
print("GloVe Sample: \n{}".format(emb_matrix_glove[0][0]))
print("Fast Text Sample: \n{}".format(emb_matrix_fast_text[0][0]))

# Create the embedding layers
emb_layer_glove = get_pretrained_embedding_layer(
    len(word_index) + 1, GLOVE_DIM, emb_matrix_glove, MAX_SEQ_LEN)
emb_layer_fast_text = get_pretrained_embedding_layer(
    len(word_index) + 1, FAST_TEXT_DIM, emb_matrix_fast_text, MAX_SEQ_LEN)

print("Done")
