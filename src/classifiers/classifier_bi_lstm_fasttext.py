from .classifier_bi_lstm import BiLstmClassifier
import tensorflow as tf
from tensorflow import keras
from src.utils import logger
import matplotlib.pyplot as plt 
from tensorflow.keras import initializers
from tensorflow.keras import regularizers 
import pandas as pd
import numpy as np


class BiLSTMClassifierFastText(BiLstmClassifier):
    def __init__(
        self,
        embedding_input_dim,
        emb_matrix,
        max_seq_len,
        name="Bi-LSTM Based Model",
        embedding_output_dim=100,
        lstm_layers=50,
        mlp_layers=16,
        mlp_activation=tf.nn.relu,
        dropout_1=0.5,
        dropout_2=0.5,
        dropout_3=0.5,
        output_activation=tf.nn.sigmoid,
        epochs=10,
        batch_size=512,
        optimizer="adam",
        loss_function="binary_crossentropy",
        metrics=["accuracy"],
        loglevel="INFO",
        logfile=None,
    ):
        self.name = name,
        self.embedding_input_dim = embedding_input_dim
        self.embedding_output_dim = embedding_output_dim
        self.lstm_layers = lstm_layers
        self.mlp_layers = mlp_layers
        self.mlp_activation = mlp_activation
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.dropout_3 = dropout_3
        self.output_activation = output_activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.logging = logger.get_logger(loglevel, logfile)
        self.logging.info("Initialized Classifier: {}".format(name))
        self.log_classifier_info()
        self.model_history = None
        self.emb_matrix = emb_matrix
        print(max_seq_len)
        print(embedding_input_dim)
        print(emb_matrix.shape)

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        save_to_file=False,
        file_name=None,
    ):
        self.logging.info("Classifier: {} - Fitting Started".format(self.name))
        self.logging.info("Train Size: {}, Validation Size: {}".format(len(X_train), len(X_val)))
        model = keras.Sequential()
        model.add(keras.layers.Embedding(
                    input_dim=self.embedding_input_dim,
                    weights=[self.emb_matrix], 
                    output_dim=self.embedding_output_dim, 
                    trainable=False))
        model.add(keras.layers.Bidirectional(
            layer=keras.layers.LSTM(units=self.lstm_layers), merge_mode="concat", input_shape=(1, 100))
        )
        model.add(keras.layers.Dropout(self.dropout_2))
        model.add(keras.layers.Dense(
            units=self.mlp_layers,
            activation=self.mlp_activation,
            kernel_initializer=initializers.glorot_normal,
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l1(0.01),

        ))
        model.add(keras.layers.Dropout(self.dropout_3))
        model.add(keras.layers.Dense(
            units=1,
            activation=self.output_activation,
            kernel_initializer=initializers.glorot_normal,
            kernel_regularizer=regularizers.l2(0.01),
            activity_regularizer=regularizers.l1(0.01),
        ))
        self.logging.info(model.summary())
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=self.metrics,
        )
        self.model_history = model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=[X_val, y_val],
            verbose=2,
        )
        return model
