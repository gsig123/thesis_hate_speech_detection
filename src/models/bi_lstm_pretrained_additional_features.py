from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout
from keras.models import Model
from keras.layers import concatenate
from keras.optimizers import Adam


def create_model(
    embedding_layer,
    lstm_units=10,
    dropout=0.2,
    dense_aux_units=16,
    dense_combined_units=16,
    dense_aux_activation="relu",
    dense_combined_activation="relu",
    dense_final_activation="sigmoid",
    optimizer="adam",
    lr=1e-3,
    loss_function="binary_crossentropy",
    metrics=["accuracy"],
    additional_features=[],
    max_seq_len=1000,
    aux_input_num_features=1,
):
    # Add more options here...
    if optimizer == "adam":
        optimizer = Adam(lr=lr)

    # Bi-LSTM Layers
    bi_lstm_input = Input(
        shape=(max_seq_len,),
        dtype="int32",
        name="bi_lstm_input",
    )
    bi_lstm = emb_layer(bi_lstm_input)
    bi_lstm = Bidirectional(
        layer=LSTM(
            units=lstm_units,
            dropout=dropout,
            recurrent_dropout=dropout,
        ),
        merge_mode="concat"
    )(bi_lstm)

    # Aux Feature Layers
    aux_input = Input(shape(aux_input_num_features,), name="aux_input")
    aux = Dense(dense_aux_units, activation=dense_aux_activation)(aux_input)
    aux = Dropout(dropout)(aux)

    # Combine the BiLSTM + Aux
    combined_input = concatenate([bi_lstm, aux])

    # Combined layers
    combined = Dense(
        dense_combined_units,
        activation=dense_combined_activation,
    )(combined_input)
    combined = Dropout(dropout)(combined)
    combined = Dense(1, activation=dense_final_activation)(combined)

    # Create and compile the model
    model = Model(inputs=[lstm_input, aux_input], outputs=combined)
    model.compile(loss=loss_function, optimizer=optimizer, metric=metrics)

    return model
