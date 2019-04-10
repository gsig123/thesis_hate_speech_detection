# === GloVe Pretrained Embeddings ====
GLOVE_DIM = 100
GLOVE_EN_PATH = "./glove_pre_trained/glove.6B.100d.txt"
# === FastText Pretrained Embeddings ===
FAST_TEXT_DIM = 300
FAST_TEXT_EN_PATH = "./fast_text_pre_trained/cc.en.300.vec"
FAST_TEXT_100_DIM = 100
FAST_TEXT_EN_PATH_100d = "./fast_text_pre_trained/cc.en.100.vec"
MODEL_PATH_FAST_TEXT_OFFENS_EVAL_EN_300d = "./models/fast_text/OffensEval_EN_300d.bin"
FAST_TEXT_EN_OFFENS_EVAL_300d = "./fast_text_pre_trained/EN_OffensEval_300d.vec"

# === Default Neural Network Model Params ===
NUM_VECTORS = 50000  # Max number of vectors to read in from FastText
# MAX_NUM_WORDS = 20000  # Max number of words to consider in dataset, ordered by frequence
# MAX_SEQ_LEN = 1000  # Max length of sequence in embedding layer
MAX_NUM_WORDS = 1000
MAX_SEQ_LEN = 100

# === Training Data ===
EN_FILE_PATH = "./data/raw/OffensEval2019/start-kit/training-v1/offenseval-training-v1.tsv"
