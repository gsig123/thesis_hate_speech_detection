from src.utils.keras_model_log import (
    create_train_dir,
    create_meta_txt,
    create_result_txt,
    create_plots_save_model,
)
from keras.layers.core import Dense, Activation, Dropout
import keras
import tensorflow as tf
from src.preprocess.offens_eval import get_X_and_ys
from src.features.keras_padded_w2i import get_padded_w2i_matrix
from src.features.embedding_matrix import (
    create_embedding_model_fasttext,
    create_embedding_model_glove,
    create_embedding_matrix,
)
from src.layers.pretrained_embedding_layer import (
    get_pretrained_embedding_layer,
)
from src.CONSTANTS import (
    GLOVE_DIM,
    GLOVE_EN_PATH,
    FAST_TEXT_DIM,
    FAST_TEXT_EN_PATH,
    NUM_VECTORS,
    MAX_NUM_WORDS,
    MAX_SEQ_LEN,
    EN_FILE_PATH,
)
from src.evaluate.metrics import *
from src.preprocess.train_test_val_split import train_test_val_split
from src.utils.save_load_keras_model import save_model
from src.plotting.confusion_matrix import plot_confusion_matrix
from src.plotting.train_val_comparison import (
    plot_train_val_accuracy,
    plot_train_val_loss,
)

# Get the data...
data = get_X_and_ys(EN_FILE_PATH)
X = data[0]
y = data[1]

# Create padded w2i matrix from X...
X, word_index = get_padded_w2i_matrix(X, MAX_NUM_WORDS, MAX_SEQ_LEN)

# Create the embedding matrices...
emb_model_glove = create_embedding_model_glove(GLOVE_EN_PATH)
emb_model_fasttext = create_embedding_model_fasttext(
    FAST_TEXT_EN_PATH, NUM_VECTORS)

emb_matrix_glove, num_oov_glove = create_embedding_matrix(
    emb_model_glove, X, GLOVE_DIM, word_index)
emb_matrix_fast_text, num_oov_fast_text = create_embedding_matrix(
    emb_model_fasttext, X, FAST_TEXT_DIM, len(word_index) + 1, word_index)
print("GloVe Sample: \n{}".format(emb_matrix_glove[0][0]))
print("Fast Text Sample: \n{}".format(emb_matrix_fast_text[0][0]))

# Create the embedding layers
emb_layer_glove = get_pretrained_embedding_layer(
    len(word_index) + 1, GLOVE_DIM, emb_matrix_glove, MAX_SEQ_LEN)
emb_layer_fast_text = get_pretrained_embedding_layer(
    len(word_index) + 1, FAST_TEXT_DIM, emb_matrix_fast_text, MAX_SEQ_LEN)

DROPOUT_AMOUNT = 0.2
REFULARIZATION_AMOUNT = 0.01

model = keras.Sequential()
model.add(emb_layer_glove)
model.add(keras.layers.Bidirectional(
    layer=keras.layers.LSTM(
        units=5,
        dropout=DROPOUT_AMOUNT,
        recurrent_dropout=DROPOUT_AMOUNT,
    ),
    merge_mode="concat",
))
model.add(keras.layers.Dropout(DROPOUT_AMOUNT))
model.add(keras.layers.Dense(
    units=4,
))
model.add(Activation(tf.nn.relu))
model.add(keras.layers.Dropout(DROPOUT_AMOUNT))
model.add(keras.layers.Dense(
    units=1,
))
model.add(Activation('sigmoid'))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['accuracy'],
)


MODEL_NAME = "test_simple_keras"
dir_path = create_train_dir(MODEL_NAME)
meta_file_path = create_meta_txt(
    dir_path,
    MODEL_NAME,
    EN_FILE_PATH,
    model,
    units=[5, 4, 1],
    dropouts=[0.2, 0.2, 0.2],
    regularizations=[],
    activation_functions=["tf.nn.relu", "sigmoid"],
    optimizer="adam",
    loss="binary_crossentropy",
    metric="acc",
    epochs=1,
    batch_size=512,
)

X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(X, y)

history = model.fit(
    X_train,
    y_train,
    epochs=1,
    batch_size=512,
    validation_data=[X_val, y_val],
    verbose=2,
)

model.evaluate(X_test, y_test)

y_mapping = data[4]

y_pred = get_y_pred_two_categories(model, X_test)

f1 = f1_score(y_test, y_pred)
recall = recall(y_test, y_pred)
precision = precision(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred, 2, ["NOT", "OFF"])

result_file_path = create_result_txt(
    dir_path,
    y_mapping,
    f1,
    recall,
    precision,
    confusion_matrix,
    len(X_train),
    len(X_val),
    len(X_test),
)

create_plots_save_model(
    dir_path,
    y_test,
    y_pred,
    y_mapping,
    history,
    model,
)
