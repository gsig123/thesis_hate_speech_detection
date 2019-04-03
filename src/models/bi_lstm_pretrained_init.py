from keras import Sequential
from keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Activation,
)
from keras.optimizers import SGD


def create_model(
    embedding_layer,
    lstm_units=5,
    dropout_1=0.2,
    dropout_2=0.2,
    dropout_3=0.2,
    recurrent_dropout=0.2,
    dense_1_units=4,
    dense_2_units=1,
    dense_1_activation="relu",
    dense_2_activation="sigmoid",
    optimizer="adam",
    loss_function="binary_crossentropy",
    metrics=["accuracy"],
):
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(
        layer=LSTM(
            units=lstm_units,
            dropout=dropout_1,
            recurrent_dropout=recurrent_dropout,
        ),
        merge_mode="concat",
    ))
    model.add(Dropout(dropout_2))
    model.add(Dense(units=dense_1_units))
    model.add(Activation(dense_1_activation))
    model.add(Dropout(dropout_3))
    model.add(Dense(units=dense_2_units))
    model.add(Activation(dense_2_activation))
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics,
    )
    return model
