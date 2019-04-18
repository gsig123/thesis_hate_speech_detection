from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import *
import re


def tokenize(sentence, language="english", stem=False):
    """
    Input: a sentence and the language, language is a string, f.x. "english"
    Output: A list of words, stopwords and punctuation ignored
    """
    stemmer = PorterStemmer()
    sentence = simplify(sentence)
    sentence = remove_emojis(sentence)
    stop_words = list(set(stopwords.words(language)))
    stop_words.extend(list(string.punctuation))
    stop_words.extend(["’"])
    stop_words.extend(["#ff", "ff", "rt"])  # Twitter stuff
    tokenized = [
        i for i in word_tokenize(sentence.lower()) if i not in stop_words
    ]
    if stem:
        tokenized = [stemmer.stem(i) for i in tokenized]
    return tokenized


def simplify(text_string):
    """
    Input: A string
    Output: The string where URLS have been replaced by URLHERE,
            twitter mentions have been replaced by MENTIONHERE,
            and hashtags have been replaced by HASHTAGHERE.
    """
    space_pattern = "\s+"
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return parsed_text


def remove_emojis(my_string):

    emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', my_string)  # no emoji

