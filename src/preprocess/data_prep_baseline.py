from .data_prep_hsaofl import DataPrepHSAOFL
from nltk.stem.porter import *
import nltk


class DataPrepBaseline(DataPrepHSAOFL):
    """
    TODO: POS Tagging for other languages than English (get_token_matrix_for_pos_tags)
    TODO: Sentiment Analysis for other languages than English (get_other_features)
    """

    def __init__(self, language):
        self.stemmer = PorterStemmer()
        self.stopwords = nltk.corpus.stopwords.words(language)
        # Twitter stuff;
        # rt = retweet
        # #ff / ff = follow friday
        self.stopwords.extend(["#ff", "ff", "rt"])
