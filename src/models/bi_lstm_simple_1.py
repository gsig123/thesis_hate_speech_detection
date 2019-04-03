from keras import Sequential
from keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
)
from keras.optimizers import SGD


def create_model(
    emb_input_dim,  # len(w2i_dict)
    emb_output_dim=300,
    lstm_units=50,
    dense_1_units=64,
    dense_1_activation="relu",
    dropout_1=0.1,
    dropout_2=0.1,
    dense_2_units=1,
    dense_2_activation="sigmoid",
    learning_rate=0.5,
    loss_function="binary_crossentropy",
    metrics=["accuracy"],
):
    model = Sequential()
    model.add(Embedding(
        input_dim=emb_input_dim,
        output_dim=emb_output_dim,
    ))
    model.add(Bidirectional(
        layer=LSTM(
            units=lstm_units,
        ),
        merge_mode="concat",
    ))
    model.add(Dropout(dropout_1))
    model.add(Dense(
        units=dense_1_units,
        activation=dense_1_activation,
    ))
    model.add(Dropout(dropout_2))
    model.add(Dense(
        units=dense_2_units,
        activation=dense_2_activation,
    ))
    model.compile(
        optimizer=SGD(lr=learning_rate),
        loss=loss_function,
        metrics=metrics,
    )
    return model