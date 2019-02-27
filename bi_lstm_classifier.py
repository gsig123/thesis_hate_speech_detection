from src.preprocess.data_prep_offenseval import DataPrepOffensEval
from src.classifiers.classifier_bi_lstm import BiLstmClassifier
from src.feature_extraction.w2i import w2i
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bi-LSTM Based Classifier")
    parser.add_argument("--train-file",
                        help="File path for the training set",
                        type=str,
                        default="./data/raw/OffensEval2019/start-kit/training-v1/offenseval-training-v1.tsv",
                        )
    parser.add_argument("--test-file",
                        help="File path for the test set",
                        type=str,
                        )
    parser.add_argument("--lstm-layers",
                        help="LSTM Hidden Layers",
                        type=int,
                        default=50,
                        )
    parser.add_argument("--mlp-layers",
                        help="MLP Hidden Layers",
                        type=int,
                        default=16,
                        )
    parser.add_argument("--epochs",
                        help="Number of training Epochs",
                        type=int,
                        default=10,
                        )
    parser.add_argument("--batch-size",
                        help="Batch Size",
                        type=int,
                        default=512,
                        )
    parser.add_argument("--logfile",
                        help="File path for log file",
                        type=str,
                        default="logfile.txt",
                        )
    parser.add_argument("--train-val-loss-file",
                        help="File path for plot of Train/Val Loss",
                        type=str,
                        default="train-val-loss.png",
                        )
    parser.add_argument("--train-val-accuracy-file",
                        help="File path for plot of Train/Val Accuracy",
                        type=str,
                        default="train-val-accuracy.png",
                        )
    parser.add_argument("--confusion-plot-file",
                        help="File path for plot of confusion matrix",
                        type=str,
                        default="confusion-matrix.png",
                        )
    args = parser.parse_args()

    # Get user defined variables:
    train_file_path = args.train_file
    test_file_path = args.test_file
    log_file_path = args.logfile
    train_val_loss_file_path = args.train_val_loss_file
    train_val_acc_file_path = args.train_val_accuracy_file
    confusion_plot_file_path = args.confusion_plot_file
    lstm_layers = args.lstm_layers
    mlp_layers = args.mlp_layers
    epochs = args.epochs
    batch_size = args.batch_size
    # Get the training data:
    dp = DataPrepOffensEval()
    result_tuple = dp.get_X_and_ys(file_path=train_file_path)
    X = result_tuple[0]
    y_sub_a = result_tuple[1]
    sub_a_mapping = result_tuple[4]
    # Transform the training data into w2i form:
    X, w2i_dict, i2w_dict = w2i(X)
    # Create a train/test set with 80%/20% of the data
    X_train, X_test, y_sub_a_train, y_sub_a_test = dp.train_test_split(
        X, y_sub_a, test_size=0.2)
    # Create validation set, 10% of train data
    X_train, X_val, y_sub_a_train, y_sub_a_val = dp.train_test_split(
        X_train, y_sub_a_train, test_size=0.1,
    )
    # Create Classifier Instance:
    classifier = BiLstmClassifier(
        embedding_input_dim=len(w2i_dict),
        epochs=epochs,
        batch_size=batch_size,
        lstm_layers=lstm_layers,
        mlp_layers=mlp_layers,
        logfile=log_file_path,
    )
    # Train the model
    model = classifier.fit(X_train, y_sub_a_train, X_val, y_sub_a_val)
    # Plot train/val loss and accuracy to evaluate overfitting
    classifier.plot_train_val_loss(file_path=train_val_loss_file_path)
    classifier.plot_train_val_metric(file_path=train_val_acc_file_path)
    y_test_pred = classifier.predict(X_test, model)
    classifier.log_metrics(y_sub_a_test, y_test_pred, sub_a_mapping)
    class_names = list(sub_a_mapping.keys())
    confusion_df = classifier.confusion_matrix(
        y_sub_a_test, y_test_pred, 2, class_names)
    classifier.plot_confusion_matrix(
        confusion_df, file_path=confusion_plot_file_path)
