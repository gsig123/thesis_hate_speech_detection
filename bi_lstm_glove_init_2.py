from src.models.classifier import Classifier
from src.models.bi_lstm_pretrained_init import create_model
from src.CONSTANTS import (
    GLOVE_DIM,
    GLOVE_EN_PATH,
    NUM_VECTORS,
    MAX_NUM_WORDS,
    MAX_SEQ_LEN,
    EN_FILE_PATH,
)
from src.features.embedding_matrix import (
    get_embedding_model_glove,
    get_embedding_matrix,
)
from src.layers.pretrained_embedding_layer import (
    get_pretrained_embedding_layer,
)
from src.preprocess.offens_eval import get_X_and_ys
from src.features.keras_padded_w2i import get_padded_w2i_matrix


LSTM_UNITS = 10
DROPOUT_1 = 0.2
DROPOUT_2 = 0.2
DROPOUT_3 = 0.2
RECURRENT_DROPOUT = 0.2
DENSE_1_UNITS = 4
DENSE_2_UNITS = 1
DENSE_1_ACTIVATION = "relu"
DENSE_2_ACTIVATION = "sigmoid"
OPTIMIZER = "adam"
LOSS_FUNCTION = "binary_crossentropy"
METRICS = ["accuracy"]

MODEL_NAME = "BiLSTM Initialized w. GloVe (2)"
EPOCHS = 20
BATCH_SIZE = 512
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 0
TRAIN_FILE_PATH = EN_FILE_PATH


if __name__ == "__main__":
    data = get_X_and_ys(TRAIN_FILE_PATH)
    X = data[0]
    y = data[1]
    y_mapping = data[4]

    X, word_index = get_padded_w2i_matrix(X, MAX_NUM_WORDS, MAX_SEQ_LEN)
    emb_model_glove = get_embedding_model_glove(
        GLOVE_EN_PATH,
    )
    emb_matrix_glove, num_oov_glove = get_embedding_matrix(
        emb_model_glove,
        GLOVE_DIM,
        word_index,
    )

    emb_layer_glove = get_pretrained_embedding_layer(
        len(word_index) + 1,
        GLOVE_DIM,
        emb_matrix_glove,
        MAX_SEQ_LEN,
    )

    model = create_model(
        embedding_layer=emb_layer_glove,
        lstm_units=LSTM_UNITS,
        dropout_1=DROPOUT_1,
        dropout_2=DROPOUT_2,
        dropout_3=DROPOUT_3,
        recurrent_dropout=RECURRENT_DROPOUT,
        dense_1_units=DENSE_1_UNITS,
        dense_2_units=DENSE_2_UNITS,
        dense_1_activation=DENSE_1_ACTIVATION,
        dense_2_activation=DENSE_2_ACTIVATION,
        optimizer=OPTIMIZER,
        loss_function=LOSS_FUNCTION,
        metrics=METRICS,
    )
    clf = Classifier(
        model_name=MODEL_NAME,
        units=[LSTM_UNITS, DENSE_1_UNITS, DENSE_2_UNITS],
        dropouts=[DROPOUT_1, RECURRENT_DROPOUT, DROPOUT_2, DROPOUT_3],
        regularizations=[],
        activation_functions=[DENSE_1_ACTIVATION, DENSE_2_ACTIVATION],
        optimizer=OPTIMIZER,
        loss=LOSS_FUNCTION,
        metric=METRICS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model=model,
        X=X,
        y=y,
        y_mapping=y_mapping,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        train_file_path=TRAIN_FILE_PATH,
        num_oov_words=num_oov_glove,
    )

    clf.train()
    clf.predict()
