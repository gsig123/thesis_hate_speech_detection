from .classifier_base import Classifier
import tensorflow as tf
from tensorflow import keras
from src.utils import logger
import matplotlib.pyplot as plt


class BiLstmClassifier(Classifier):

    def __init__(
        self,
        embedding_input_dim,
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

    def log_classifier_info(self):
        self.logging.info("LSTM Layers: {}".format(self.lstm_layers))
        self.logging.info("Embedding Input Dimension: {}".format(
            self.embedding_input_dim))
        self.logging.info("Embedding Output Dimension: {}".format(
            self.embedding_output_dim))
        self.logging.info("MLP Layers: {}".format(self.mlp_layers))
        self.logging.info("MLP Activation: {}".format(self.mlp_activation))
        self.logging.info("Dropout 1: {}".format(self.dropout_1))
        self.logging.info("Dropout 2: {}".format(self.dropout_2))
        self.logging.info("Dropout 3: {}".format(self.dropout_3))
        self.logging.info("Output Activation: {}".format(
            self.output_activation))
        self.logging.info("Epochs: {}".format(self.epochs))
        self.logging.info("Batch Size: {}".format(self.batch_size))
        self.logging.info("Optimizer: {}".format(self.optimizer))
        self.logging.info("Loss Function: {}".format(self.loss_function))
        self.logging.info("Metrics: {}".format(self.metrics))

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        save_to_file=False,
        file_name=None
    ):
        self.logging.info("Classifier: {} - Fitting Started".format(self.name))
        self.logging.info("Train Size: {}, Validation Size: {}".format(
            len(X_train), len(X_val)))
        model = keras.Sequential()
        model.add(keras.layers.Embedding(
            input_dim=self.embedding_input_dim,
            output_dim=self.embedding_output_dim,
        ))
        model.add(keras.layers.Dropout(self.dropout_1))
        model.add(keras.layers.Bidirectional(
            layer=keras.layers.LSTM(self.lstm_layers),
            merge_mode="concat")
        )
        model.add(keras.layers.Dropout(self.dropout_2))
        model.add(keras.layers.Dense(
            units=self.mlp_layers,
            activation=self.mlp_activation,
        ))
        model.add(keras.layers.Dropout(self.dropout_3))
        model.add(keras.layers.Dense(
            units=1,
            activation=self.output_activation,
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

    def plot_train_val_loss(self, file_path=None):
        if not self.model_history:
            self.logging.error("Model History Doesn't Exist")
            return
        plt.clf()
        history_dict = self.model_history.history
        train_loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, 'r', label='Training Loss')
        plt.plot(epochs, val_loss, 'b', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if file_path:
            plt.savefig(file_path)
        else:
            plt.show()

    def plot_train_val_metric(self, metric='acc', file_path=None):
        if not self.model_history:
            self.logging.error("Model History Doesn't Exist")
            return
        plt.clf()
        history_dict = self.model_history.history
        train_metric = history_dict[metric]
        val_metric = history_dict['val_' + metric]
        epochs = range(1, len(train_metric) + 1)
        plt.plot(epochs, train_metric, 'r', label='Training {}'.format(metric))
        plt.plot(epochs, val_metric, 'b', label='Validation {}'.format(metric))
        plt.title('Training and Validation {}'.format(metric))
        plt.xlabel('Epochs')
        plt.ylabel('{}'.format(metric))
        plt.legend()
        if file_path:
            plt.savefig(file_path)
        else:
            plt.show()

    def log_metrics(self, y_true, y_pred, y_map):
        recall = self.recall(y_true, y_pred)
        precision = self.precision(y_true, y_pred)
        f1_score = self.f1_score(y_true, y_pred)
        accuracy = self.precision(y_true, y_pred)
        self.logging.info("Index Map: {}".format(y_map))
        self.logging.info("Recall: {}".format(recall))
        self.logging.info("Precision: {}".format(precision))
        self.logging.info("F1 Score: {}".format(f1_score))
        self.logging.info("Accuracy: {}".format(accuracy))

    def predict(self, X, model):
        y_pred = model.predict(X)
        y_pred_binary = [0 if value[0] < 0.5 else 1 for value in y_pred]
        return y_pred_binary
