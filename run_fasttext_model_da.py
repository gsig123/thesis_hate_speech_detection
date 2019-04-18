import sys
import click
import pandas as pd
from src.models.classifier import Classifier
from src.models.bi_lstm_pretrained_init import create_model
from src.CONSTANTS import (
    MAX_NUM_WORDS,
    MAX_SEQ_LEN,
    EN_FILE_PATH,
    DA_FILE_PATH,
    EN_DA_FILE_PATH,
    EN_DA_FILE_PATH,
    EN_EMB_FILE_PATH,
    DA_EMB_FILE_PATH,
    EN_DA_EMB_FILE_PATH,
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
@click.option("--train_file_path", default=EN_FILE_PATH, help="Path to train file")
@click.option("--embedding_file_path", default=EN_EMB_FILE_PATH, help="Path to embedding file")
@click.option("--max_seq_len", default=MAX_SEQ_LEN, help="Max sequence length in input (number of words in sentence)")
@click.option("--max_num_words", default=MAX_NUM_WORDS, help="Max # words to consider in tokenization")
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
    train_file_path,
    embedding_file_path,
    max_seq_len,
    max_num_words,
):
    arguments = sys.argv
    en_data = get_X_and_ys(EN_FILE_PATH)
    da_data = get_X_and_ys(DA_FILE_PATH)
    X_en = en_data[0]
    X_da = da_data[0]
    X_original_en = X_en
    X_original_da = X_da
    y_en = en_data[1]
    y_da = da_data[1]
    y_mapping_en = en_data[4]
    y_mapping_da = da_data[4]
    X_en, word_index_en = get_padded_w2i_matrix(X_en, max_num_words, max_seq_len)
    X_da, word_index_da = get_padded_w2i_matrix(X_da, max_num_words, max_seq_len)

    en_da_data = get_X_and_ys(EN_DA_FILE_PATH)
    X_en_da = en_da_data[0]
    X_en_da, word_index_en_da = get_padded_w2i_matrix(X_en_da, max_seq_len, max_num_words)

    emb_dim = FAST_TEXT_DIM
    
    emb_model = create_embedding_model(EN_DA_EMB_FILE_PATH)
    
    emb_matrix, num_oov = create_embedding_matrix(
        emb_model,
        emb_dim,
        word_index_en_da,
    )
    
    print("# OOV: {}".format(num_oov))

    emb_layer = get_pretrained_embedding_layer(
        len(word_index_en_da) + 1,
        emb_dim,
        emb_matrix,
        max_seq_len,
    )

    dense_2_units = 1

    model = create_model(
        embedding_layer=emb_layer,
        lstm_units=lstm_units,
        dropout_1=dropout,
        dropout_2=dropout,
        dropout_3=dropout,
        recurrent_dropout=dropout,
        dense_1_units=dense_units,
        dense_2_units=dense_2_units,
        dense_1_activation=dense_1_activation,
        dense_2_activation=dense_2_activation,
        optimizer=optimizer,
        loss_function=loss_function,
        metrics=METRICS,
    )
    clf = Classifier(
        arguments=arguments,
        model_name=model_name,
        units=[lstm_units, dense_units, dense_2_units],
        dropouts=[dropout, dropout, dropout, dropout],
        regularizations=[],
        activation_functions=[dense_1_activation, dense_2_activation],
        optimizer=optimizer,
        loss=loss_function,
        metric=METRICS,
        epochs=epochs,
        batch_size=batch_size,
        model=model,
        X=pd.DataFrame(X_en),
        X_original=pd.DataFrame(X_original),
        y=pd.DataFrame(y),
        y_mapping=y_mapping,
        test_size=test_size,
        val_size=val_size,
        random_state=RANDOM_STATE,
        train_file_path=train_file_path,
        num_oov_words=num_oov,
        max_seq_len=max_seq_len,
        max_num_words=max_num_words,
    )

    clf.train()
    clf.predict()

if __name__ == "__main__":
    main()
