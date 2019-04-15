from keras import Sequential
from keras.models import Model
from keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Activation,
    Input,
)
from keras.optimizers import SGD
from keras.layers import Add


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
    additional_features=[],
    max_seq_len=1000,
):
    input = Input(shape=(max_seq_len,))
    embedding = embedding_layer(input)
    bi_lstm_layer = Bidirectional(
        layer=LSTM(
            units=lstm_units,
            dropout=dropout_1,
            recurrent_dropout=recurrent_dropout,
        ),
        merge_mode="concat",
    )(embedding)
    emb_features = Dropout(dropout_2)(bi_lstm_layer)

    # Adding hand-picked features
    nb_features = len(additional_features[0])
    other_features = Input(shape=(nb_features,))

    model_final = Add()([input, other_features])

    model_final = Dense(units=dense_1_units)(model_final)
    model_final = Activation(dense_1_activation)(model_final)
    model_final = Dropout(dropout_3)(model_final)
    model_final = Dense(units=dense_2_units)(model_final)
    model_final = Activation(dense_2_activation)(model_final)

    model_final = Model([input, other_features], model_final)
    
    model_final.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics,
    )

    return model_final
