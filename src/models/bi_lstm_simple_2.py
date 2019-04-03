from keras import Sequential
from keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
)
from keras.optimizers import SGD
from keras.initializers import glorot_normal


def create_model(
    emb_input_dim,  # len(w2i_dict)
    emb_output_dim=20,
    lstm_units=10,
    dense_1_units=4,
    dense_1_activation="relu",
    dropout_1=0.5,
    dropout_2=0.1,
    dropout_3=0.1,
    dropout_4=0.1,
    dropout_5=0.1,
    dense_2_units=1,
    dense_2_activation="sigmoid",
    loss_function="binary_crossentropy",
    metrics=["accuracy"],
    optimizer="adam",
):
    model = Sequential()
    model.add(Embedding(
        input_dim=emb_input_dim,
        output_dim=emb_output_dim,
        embeddings_initializer=glorot_normal(seed=None),
    ))
    model.add(Dropout(dropout_1))
    model.add(Bidirectional(
        layer=LSTM(
            units=lstm_units,
            dropout=dropout_2,
            recurrent_dropout=dropout_3,
        ),
        merge_mode="concat",
    ))
    model.add(Dropout(dropout_4))
    model.add(Dense(
        units=dense_1_units,
        activation=dense_1_activation,
    ))
    model.add(Dropout(dropout_5))
    model.add(Dense(
        units=dense_2_units,
        activation=dense_2_activation,
    ))
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics,
    )
    return model