from sklearn.feature_extraction.text import TfidfTransformer


def tfidf_from_bow(bow_array, smooth_idf=False):
    """
    Input: Bag of words array (from ./bag_of_words.py)
    Output: All values in the bow array have been replaced by the
            tf-idf score.
    """
    transformer = TfidfTransformer(smooth_idf=smooth_idf)
    tfidf = transformer.fit_transform(bow_array)
    return tfidf.toarray()
