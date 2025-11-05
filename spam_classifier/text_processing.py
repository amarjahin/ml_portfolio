import re
from collections import Counter


def clean_text(text):
    """
    For a given text, return a list of words that are not digits. 
    This also removes punctuation at the end of words.
    It is case sensitive, because it can be indicative of spam or not. 
    Parameters:
        text (str): The text to clean
    Returns:
        list: A list of words
    """
    return [word for word in re.findall(r'\b\w+\b', text) if not word.isdigit()]

def create_vocab(df):
    """
    Create a set of unique words from the text in the dataframe.
    Parameters:
        df (pandas.DataFrame): The dataframe containing the text
    Returns:
        set: A set of unique words
    """
    vocab = {word for text in df["text"] 
            for word in clean_text(text)}
    return vocab
    
def word_counts_by_label(df):
    """
    Count the number of times each word appears in the text for each label.
    Parameters:
        df (pandas.DataFrame): The dataframe containing the text
    Returns:
        dict: A dictionary with the word counts for each label
    """
    # counts = {
    #     "spam": Counter(),
    #     "ham": Counter()
    # }
    counts = {label: Counter() for label in df["label"].unique()}

    for _, row in df.iterrows():
        label = row["label"]
        text = row["text"]
        counts[label].update(word for word in clean_text(text))
    return counts



