from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS


def sentiment_score_english(text):
    """
    Input: A text string
    Output: -1 for negative, 0 for neutral, 1 for positive
    """
    sentiment_analyzer = VS()
    sentiment = sentiment_analyzer.polarity_scores(text)
    compound = sentiment["compound"]
    # Neg/Neu/Pos based on the documentation
    if compound >= 0.05:
        return 1
    elif compound >= -0.05:
        return 0
    else:
        return -1
