from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from .data_prep_base import DataPrep
import numpy as np
import pandas as pd


class DataPrepHSAOFL(DataPrep):

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stopwords = nltk.corpus.stopwords.words("english")
        self.stopwords.extend(["#ff", "ff", "rt"])

    def preprocess(self, text_string):
        """
        Accepts a text string and replaces:
        1) urls with URLHERE
        2) lots of whitespace with one instance
        3) mentions with MENTIONHERE

        This allows us to get standardized counts of urls and mentions
        Without caring about specific people mentioned
        """
        space_pattern = '\s+'
        giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                           '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        mention_regex = '@[\w\-]+'
        parsed_text = re.sub(space_pattern, ' ', text_string)
        parsed_text = re.sub(giant_url_regex, '', parsed_text)
        parsed_text = re.sub(mention_regex, '', parsed_text)
        return parsed_text

    def tokenize(self, tweet):
        """
        Removes punctuation & excess whitespace, sets to lowercase,
        and stems tweets. Returns a list of stemmed tokens.
        """
        tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
        tokens = [self.stemmer.stem(t) for t in tweet.split()]
        return tokens

    def basic_tokenize(self, tweet):
        """
        Same as tokenize but without the stemming
        """
        tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
        return tweet.split()

    def get_tfidf_matrix(self, tweets):
        vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize,
            preprocessor=self.preprocess,
            ngram_range=(1, 3),
            stop_words=self.stopwords,
            use_idf=True,
            smooth_idf=False,
            norm=None,
            decode_error='replace',
            max_features=10000,
            min_df=5,
            max_df=0.75
        )
        # Construct tfid matrix and get relevan scores
        tfidf = vectorizer.fit_transform(tweets).toarray()
        vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names())}
        idf_vals = vectorizer.idf_
        # keys are indices; values are IDF scores
        idf_dict = {i: idf_vals[i] for i in vocab.values()}
        return tfidf, vocab, idf_vals, idf_dict

    def get_pos_tags_as_strings(self, tweets):
        # Get POS tags for tweets and save as a string
        tweet_tags = []
        for t in tweets:
            tokens = self.basic_tokenize(self.preprocess(t))
            tags = nltk.pos_tag(tokens)
            tag_list = [x[1] for x in tags]
            tag_str = " ".join(tag_list)
            tweet_tags.append(tag_str)
        return tweet_tags

    def get_token_matrix_for_pos_tags(self, tweet_tags):
        # We can use the TFID vectorizer to get a token matrix for the POS tags
        pos_vectorizer = TfidfVectorizer(
            tokenizer=None,
            lowercase=False,
            preprocessor=None,
            ngram_range=(1, 3),
            stop_words=None,
            use_idf=False,
            smooth_idf=False,
            norm=None,
            decode_error='replace',
            max_features=5000,
            min_df=5,
            max_df=0.75,
        )
        # Construct POS TF matrix and get vocab dict
        pos = pos_vectorizer.fit_transform(tweet_tags).toarray()
        pos_vocab = {v: i for i, v in enumerate(
            pos_vectorizer.get_feature_names())}
        return pos, pos_vocab

    def count_twitter_objs(self, text_string):
        """
        Accepts a text string and replaces:
        1) urls with URLHERE
        2) lots of whitespace with one instance
        3) mentions with MENTIONHERE
        4) hashtags with HASHTAGHERE

        This allows us to get standardized counts of urls and mentions
        Without caring about specific people mentioned.

        Returns counts of urls, mentions, and hashtags.
        """
        space_pattern = '\s+'
        giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                           '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        mention_regex = '@[\w\-]+'
        hashtag_regex = '#[\w\-]+'
        parsed_text = re.sub(space_pattern, ' ', text_string)
        parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
        parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
        parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
        return (
            parsed_text.count('URLHERE'),
            parsed_text.count('MENTIONHERE'),
            parsed_text.count('HASHTAGHERE')
        )

    def get_other_features(self, tweet):
        """
        This function takes a string and returns a list of features.
        These include Sentiment scores, Text and Readability scores,
        as well as Twitter specific features
        """
        sentiment_analyzer = VS()
        sentiment = sentiment_analyzer.polarity_scores(tweet)
        words = self.preprocess(tweet)  # Get text only
        syllables = textstat.syllable_count(words)
        num_chars = sum(len(w) for w in words)
        num_chars_total = len(tweet)
        num_terms = len(tweet.split())
        num_words = len(words.split())
        avg_syl = round(
            float((syllables + 0.001)) / float(num_words + 0.001), 4)
        num_unique_terms = len(set(words.split()))
        # Modified FK grade, where avg words per sentence is just num words/1
        FKRA = round(float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)
        # Modified FRE score, where sentence fixed to 1
        FRE = round(206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)), 2)
        twitter_objs = self.count_twitter_objs(tweet)
        retweet = 0
        if "rt" in words:
            retweet = 1
        features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total,
                    num_terms, num_words, num_unique_terms, sentiment['neg'],
                    sentiment['pos'], sentiment['neu'], sentiment['compound'],
                    twitter_objs[2], twitter_objs[1], twitter_objs[0], retweet]
        # features = pandas.DataFrame(features)
        return features

    def get_other_features_array(self, tweets):
        feats = []
        for t in tweets:
            feats.append(self.get_other_features(t))
        other_features_names = ["FKRA", "FRE", "num_syllables",
                                "avg_syl_per_word", "num_chars",
                                "num_chars_total", "num_terms",
                                "num_words", "num_unique_words",
                                "vader neg", "vader pos",
                                "vader neu", "vader compound",
                                "num_hashtags", "num_mentions",
                                "num_urls", "is_retweet"]
        return np.array(feats), other_features_names

    def get_X_y_feature_names(self, dataset, tweet_column_name, y_column_name):
        # # Testing on first 1000 rows to speed things up
        # dataset = dataset.head(1000)
        tweets = dataset[tweet_column_name]
        tfidf, vocab, idf_vals, idf_dict = self.get_tfidf_matrix(tweets)
        tweet_tags = self.get_pos_tags_as_strings(tweets)
        pos, pos_vocab = self.get_token_matrix_for_pos_tags(tweet_tags)
        other_features, other_features_names = self.get_other_features_array(tweets)
        X = np.concatenate([tfidf, pos, other_features], axis=1)
        X = pd.DataFrame(X)

        # Get list of feature names
        feature_names = variables = [''] * len(vocab)
        for k, v in vocab.items():
            variables[v] = k
        pos_variables = [''] * len(pos_vocab)
        for k, v in pos_vocab.items():
            pos_variables[v] = k
        feature_names = variables + pos_variables + other_features_names

        y = dataset[y_column_name].astype(int)

        return X, y, feature_names
