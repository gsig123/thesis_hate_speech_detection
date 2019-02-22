from nltk import word_tokenize
from nltk.corpus import stopwords
import string


def tokenize(sentence, language="english"):
    """
    Input: a sentence and the language, language is a string, f.x. "english"
    Output: A list of words, stopwords and punctuation ignored
    """
    stop_words = list(set(stopwords.words(language)))
    stop_words.extend(list(string.punctuation))
    tokenized = [
        i for i in word_tokenize(sentence.lower()) if i not in stop_words
    ]
    return tokenized
