from gensim.models import Word2Vec
from .tokenize import tokenize


def w2v_embeddings_from_sentences(sentences,
                                  language="english",
                                  min_count=1,
                                  size=100):
    """
    Input:
    - sentences: list of sentences
    - language: language used for stopwords
    - min_count: min frequency of words
    - size: size of word embedding vectors
    Output: List of list of numpy arrays where each list represents a
            sentence, and each numpy array contains a word embedding
            for a word in the sentence.
    """
    tokenized_sentences = [tokenize(s, language) for s in sentences]
    model = Word2Vec(min_count=min_count, size=size)
    model.build_vocab(tokenized_sentences)
    model.train(
        tokenized_sentences,
        total_examples=model.corpus_count,
        epochs=model.iter
    )
    w2v_sentences = []
    for sentence in tokenized_sentences:
        w2v_sentences.append([model.wv[word] for word in sentence])
    return w2v_sentences
