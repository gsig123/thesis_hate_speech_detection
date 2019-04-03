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


class Classifier:
    def __init__(
        self,
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
        y,
        y_mapping,
        test_size,
        val_size,
        random_state,
        train_file_path,
    ):
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
        self.y = y
        self.y_mapping = y_mapping
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.train_file_path = train_file_path
        # LOG META DATA
        self.dir_path = create_train_dir(self.model_name)
        self.meta_file_path = create_meta_txt(
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
        )
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

    def predict(self):
        self.y_pred = get_y_pred_two_categories(self.model, self.X_test)
        self.f1 = f1_score(self.y_test, self.y_pred)
        self.recall = recall(self.y_test, self.y_pred)
        self.precision = precision(self.y_test, self.y_pred)
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
