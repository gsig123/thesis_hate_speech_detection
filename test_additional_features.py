from src.CONSTANTS import (
    MAX_NUM_WORDS,
    MAX_SEQ_LEN,
    EN_FILE_PATH,
    EN_EMB_FILE_PATH,
    FAST_TEXT_DIM,
)

from src.features.embedding_matrix import (
    create_embedding_model,
    create_embedding_matrix,
)

from src.layers.pretrained_embedding_layer import (
    get_pretrained_embedding_layer,
)

from src.preprocess.offens_eval import get_X_and_ys
from src.features.keras_padded_w2i import get_padded_w2i_matrix

from src.models.bi_lstm_pretrained_additional_features import create_model

RANDOM_STATE = 0
METRICS = ['accuracy']
LSTM_UNITS = 5
DENSE_UNITS = 4
DROPOUT = 0.2
DENSE_1_ACTIVATION = "relu"
DENSE_2_ACTIVATION = "sigmoid"
OPTIMIZER = "adam"
LOSS_FUNCTION = "binary_crossentropy"
EPOCHS = 1
BATCH_SIZE = 512
TEST_SIZE = 0.2
VAL_SIZE = 0.1

data = get_X_and_ys(EN_FILE_PATH)
X = data[0]
X_original = X
y = data[1]
y_mapping = data[4]
X, word_index = get_padded_w2i_matrix(X, MAX_NUM_WORDS, MAX_SEQ_LEN)
additional_features = [[1]] * len(X)


emb_model = create_embedding_model(EN_EMB_FILE_PATH)

emb_matrix, num_oov = create_embedding_matrix(
    emb_model,
    FAST_TEXT_DIM,
    word_index,
)

emb_layer = get_pretrained_embedding_layer(
    len(word_index) + 1,
    FAST_TEXT_DIM,
    emb_matrix,
    MAX_SEQ_LEN,
)

model = create_model(
    emb_layer,
    LSTM_UNITS,
    dropout_1=DROPOUT,
    dropout_2=DROPOUT,
    dropout_3=DROPOUT,
    recurrent_dropout=DROPOUT,
    dense_1_units=4,
    dense_2_units=1,
    dense_1_activation=DENSE_1_ACTIVATION,
    dense_2_activation=DENSE_2_ACTIVATION,
    optimizer=OPTIMIZER,
    loss_function=LOSS_FUNCTION,
    metrics=METRICS,
    max_seq_len=MAX_SEQ_LEN,
    additional_features=additional_features,
)
