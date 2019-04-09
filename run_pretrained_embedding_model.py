import click
import pandas as pd
from src.models.classifier import Classifier
from src.models.bi_lstm_pretrained_init import create_model
from src.CONSTANTS import (
    NUM_VECTORS,
    MAX_NUM_WORDS,
    MAX_SEQ_LEN,
    EN_FILE_PATH,
    FAST_TEXT_EN_OFFENS_EVAL_300d,
    GLOVE_EN_PATH,
    FAST_TEXT_DIM,
    GLOVE_DIM,
)
from src.features.embedding_matrix import (
    get_embedding_model_fasttext,
    get_embedding_model_glove,
    get_embedding_matrix,
)
from src.layers.pretrained_embedding_layer import (
    get_pretrained_embedding_layer,
)
from src.preprocess.offens_eval import get_X_and_ys
from src.features.keras_padded_w2i import get_padded_w2i_matrix

RANDOM_STATE = 0
METRICS = ['accuracy']


@click.command()
@click.option("--lstm_units", default=5, help="# LSTM Units")
@click.option("--dense_units", default=4, help="# Dense Units")
@click.option("--dropout", default=0.2, help="Amount of Dropout")
@click.option("--dense_1_activation", default="relu", help="Dense#1 Activation Function")
@click.option("--dense_2_activation", default="sigmoid", help="Dense#2 Activation Function")
@click.option("--optimizer", default="adam", help="Adam")
@click.option("--loss_function", default="binary_crossentropy", help="Loss Function")
@click.option("--epochs", default=20, help="# of Epochs")
@click.option("--batch_size", default=512, help="Batch Size when training")
@click.option("--test_size", default=0.2, help="Test Size")
@click.option("--val_size", default=0.1, help="Validation Size")
@click.option("--model_name", help="The name of the model")
@click.option("--glove_fasttext", type=click.Choice(["FastText", "GloVe"]), help="GloVe or FastText?")
@click.option("--train_file_path", default=EN_FILE_PATH, help="Path to train file")
@click.option("")
def main(
    lstm_units,
    dense_units,
    dropout,
    dense_1_activation,
    dense_2_activation,
    optimizer,
    loss_function,
    epochs,
    batch_size,
    test_size,
    val_size,
    model_name,
    glove_fasttext,
):
    data = get_X_and_ys(EN_FILE_PATH)
    X = data[0]
    y = data[1]
    y_mapping = data[4]
    X, word_index = get_padded_w2i_matrix(X, MAX_NUM_WORDS, MAX_SEQ_LEN)

    if glove_fasttext == "FastText":
        emb_dim = FAST_TEXT_DIM
        # emb_model = get_embedding_model_fasttext(FAST_TEXT_EN_OFFENS_EVAL_300d)
        emb_model = get_embedding_model_glove(FAST_TEXT_EN_OFFENS_EVAL_300d)
    elif glove_fasttext == "GloVe":
        emb_dim = GLOVE_DIM
        emb_model = get_embedding_model_glove(GLOVE_EN_PATH)
    
    emb_matrix, num_oov = get_embedding_matrix(
        emb_model,
        X,
        emb_dim,
        word_index,
    )
    
    print("# OOV: {}".format(num_oov))

    emb_layer = get_pretrained_embedding_layer(
        len(word_index) + 1,
        emb_dim,
        emb_matrix,
        MAX_SEQ_LEN,
    )

    model = create_model(
        embedding_layer=emb_layer,
        lstm_units=lstm_units,
        dropout_1=dropout,
        dropout_2=dropout,
        dropout_3=dropout,
        recurrent_dropout=dropout,
        dense_1_units=dense_units,
        dense_2_units=1,
        dense_1_activation=dense_1_activation,
        dense_2_activation=dense_2_activation,
        optimizer=optimizer,
        loss_function=loss_function,
        metrics=METRICS,
    )
    clf = Classifier(
        model_name=model_name,
        units=[lstm_units, dense_units, dense_units],
        dropouts=[dropout, dropout, dropout, dropout],
        regularizations=[],
        activation_functions=[dense_1_activation, dense_2_activation],
        optimizer=optimizer,
        loss=loss_function,
        metric=METRICS,
        epochs=epochs,
        batch_size=batch_size,
        model=model,
        X=pd.DataFrame(X),
        y=pd.DataFrame(y),
        y_mapping=y_mapping,
        test_size=test_size,
        val_size=val_size,
        random_state=RANDOM_STATE,
        train_file_path=EN_FILE_PATH,
        num_oov_words=num_oov,
    )

    clf.train()
    clf.predict()

if __name__ == "__main__":
    main()
