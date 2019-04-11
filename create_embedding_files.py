from pyfasttext import FastText
from gensim.models import KeyedVectors

EN_TOKEN_PATH = "./data/tokens/EN_tokens.txt"
DA_TOKEN_PATH = "./data/tokens/DA_1600_tokens.txt"

EN_PRETRAINED_PATH = "./fast_text_pre_trained/cc.en.300.vec"
DA_PRETRAINED_PATH = "./fast_text_pre_trained/cc.da.300.vec"

EN_EMBEDDINGS_PATH = "./data/tokens/EN_tokens_embeddings.txt"
DA_EMBEDDINGS_PATH = "./data/tokens/DA_1600_tokens_embeddings.txt"

EN_OOV_TOKENS_PATH = "./data/tokens/EN_tokens_OOV.txt"
DA_OOV_TOKENS_PATH = "./data/tokens/DA_1600_tokens_OOV.txt"

# Only consider the top 100k most frequent embeddings
# from the pretrained embedding stuff. If we don't limit
# this to a number it will take forever to load in the model..
NUM_VECTORS = 100000

en_emb_model = KeyedVectors.load_word2vec_format(
    EN_PRETRAINED_PATH,
    limit=NUM_VECTORS,
)

da_emb_model = KeyedVectors.load_word2vec_format(
    DA_PRETRAINED_PATH,
    limit=NUM_VECTORS,
)

EN_TOKENS = []
DA_TOKENS = []

with open(EN_TOKEN_PATH, "r") as f:
    EN_TOKENS = f.read().splitlines()

with open(DA_TOKEN_PATH, "r") as f:
    DA_TOKENS = f.read().splitlines()

en_emb_file = open(EN_EMBEDDINGS_PATH, "w")
da_emb_file = open(DA_EMBEDDINGS_PATH, "w")
en_oov_file = open(EN_OOV_TOKENS_PATH, "w")
da_oov_file = open(DA_OOV_TOKENS_PATH, "w")


for token in EN_TOKENS:
    if token in en_emb_model:
        embedding_vector = en_emb_model[token]
        en_emb_file.write(token + " " + " ".join(str(v) for v in embedding_vector) + "\n")
    else:
        en_oov_file.write(token + "\n")

for token in DA_TOKENS:
    if token in da_emb_model:
        embedding_vector = da_emb_model[token]
        da_emb_file.write(token + " " + " ".join(str(v) for v in embedding_vector) + "\n")
    else:
        da_oov_file.write(token + "\n")

en_emb_file.close()
da_emb_file.close()
en_oov_file.close()
da_oov_file.close()
