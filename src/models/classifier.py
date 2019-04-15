from ..utils.keras_model_log import (
    create_train_dir,
    create_meta_txt,
    create_result_txt,
    create_plots_save_model,
)
from ..preprocess.train_test_val_split import train_test_val_split
from ..evaluate.metrics import (
    get_y_pred_two_categories,
    f1_score,
    recall,
    precision,
    confusion_matrix,
)
import pandas as pd


class Classifier:
    def __init__(
        self,
        arguments,  # Command line arguments
        model_name,
        units,  # LIST
        dropouts,  # LIST
        regularizations,  # LIST
        activation_functions,  # LIST
        optimizer,
        loss,
        metric,
        epochs,
        batch_size,
        model,
        X,
        X_original,  # Just the tweets, not embedded or anything.
        y,
        y_mapping,
        test_size,
        val_size,
        random_state,
        train_file_path,
        num_oov_words,
        max_seq_len,
        max_num_words,
    ):
        self.arguments = arguments
        self.model_name = model_name
        self.units = units
        self.dropouts = dropouts
        self.regularizations = regularizations
        self.activation_functions = activation_functions
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.X = X
        self.X_original = X_original
        self.y = y
        self.y_mapping = y_mapping
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.train_file_path = train_file_path
        self.num_oov_words = num_oov_words
        self.max_seq_len = max_seq_len
        self.max_num_words = max_num_words
        # LOG META DATA
        self.dir_path = create_train_dir(self.model_name)
        self.meta_file_path = create_meta_txt(
            self.arguments,
            self.dir_path,
            self.model_name,
            self.train_file_path,
            self.model,
            self.units,
            self.dropouts,
            self.regularizations,
            self.activation_functions,
            self.optimizer,
            self.loss,
            self.metric,
            self.epochs,
            self.batch_size,
            self.num_oov_words,
            self.max_seq_len,
            self.max_num_words,
        )
        X = pd.DataFrame(X)
        # CREATE TRAIN TEST VAL
        X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(
            X,
            y,
            test_size=self.test_size,
            val_size=self.val_size,
            random_state=self.random_state,
        )
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val

    def train(self):
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=[self.X_val, self.y_val],
            verbose=2,
        )

    def predict(self, X_test=None, y_test=None):
        if X_test:
            self.X_test = X_test
        if y_test:
            self.y_test = y_test
        self.y_pred = get_y_pred_two_categories(self.model, self.X_test)
        self.f1 = f1_score(self.y_test, self.y_pred)
        self.recall = recall(self.y_test, self.y_pred)
        self.precision = precision(self.y_test, self.y_pred)
        # Create CSV file with predicted vs actual
        self.confusion_matrix = confusion_matrix(
            self.y_test,
            self.y_pred,
            len(self.y_mapping),
            list(self.y_mapping.keys()),
        )
        result_file_path = create_result_txt(
            self.dir_path,
            self.y_mapping,
            self.f1,
            self.recall,
            self.precision,
            self.confusion_matrix,
            len(self.X_train),
            len(self.X_val),
            len(self.X_test),
        )
        create_plots_save_model(
            self.dir_path,
            self.y_test,
            self.y_pred,
            self.y_mapping,
            self.history,
            self.model,
        )
        # Create csv with predicted vs true and original tweets
        self.true_vs_predicted_to_csv(self.y_test, self.y_pred)

    def true_vs_predicted_to_csv(self, y_test, y_pred):
        # Create the path:
        file_path = self.dir_path + "/test_vs_pred.csv"
        # Create the data
        test_indices = self.X_test.index.tolist()
        X_original_test_split = self.X_original.ix[test_indices]
        df = pd.DataFrame()
        df['index'] = test_indices
        df['tweet'] = X_original_test_split[0].values
        df['y_true'] = self.y_test.values
        df['y_pred'] = self.y_pred
        y_mapping_inverse = dict(map(reversed, self.y_mapping.items()))
        df = df.replace({"y_true": y_mapping_inverse})
        df = df.replace({"y_pred": y_mapping_inverse})
        # Write to csv
        df.to_csv(file_path)
2