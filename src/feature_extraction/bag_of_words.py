from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string


def bag_of_words(list_of_sentences,
                 language="english",
                 min_df=0.0,
                 max_df=1.0,
                 max_features=None):
    """
    Input:
    - List of sentences
    - language: Language for the stopwords
    - min_df: Ignore terms that have a document frequency strictly lower than
              the given threshold.
    - max_df: Ignore terms that have a document frequency strictly higher than
              the given threshold.
    - max_features: If not None, build a vocabulary that only considers the
                    top max_features ordered by term frequency.
    Output: BOW Numpy array ignoring stopwords and punctuation.
    """
    stop_words = list(set(stopwords.words(language)))
    stop_words.extend(list(string.punctuation))
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
    )
    X = vectorizer.fit_transform(
        list_of_sentences,
    )
    return X.toarray(), vectorizer.get_feature_names()
