import numpy as np
import pandas as pd
from text_processing import create_vocab, word_counts_by_label, clean_text


data_train = pd.read_csv('spam_classifier/data/spam_texts_train.tsv', sep='\t', 
                    header=None, names=["label", "text"])

data_valid = pd.read_csv('spam_classifier/data/spam_texts_valid.tsv', sep='\t', 
                    header=None, names=["label", "text"])


class NaiveBayes():
    """
    Naive Bayes classifier for spam detection.
    Parameters:
        df (pandas.DataFrame): The dataframe containing the training data
    """
    def __init__(self, df):
        self.vocab = create_vocab(df)
        self.counts = word_counts_by_label(df)
        # conditional probabilities of words given the label
        self.cond_props = self.conditional_probabilities(self.counts)
        # prior probability of spam
        self.phi = np.mean(df["label"] == "spam")

    def conditional_probabilities(self,counts):
        """
        Calculate the conditional probabilities of words given the label.
        Parameters:
            counts (dict): word counts by label
        Returns:
            dict: conditional probabilities of words given the label
        """
        total_spam = sum(counts["spam"].values())
        total_ham = sum(counts["ham"].values())
        probabilities = {
            "spam": {word: count / total_spam for word, count in counts["spam"].items()},
            "ham": {word: count / total_ham for word, count in counts["ham"].items()}}   
        # add a small probability to avoid log(0)
        for vocab in self.vocab:
            if not vocab in probabilities["spam"]:
                probabilities["spam"][vocab] = 1e-16
            if not vocab in probabilities["ham"]:
                probabilities["ham"][vocab] = 1e-16
        # normalize the probabilities to sum to 1
        norm_spam = sum(probabilities["spam"].values())
        norm_ham = sum(probabilities["ham"].values())
        probabilities["spam"] = {word: prob / norm_spam for word, prob in probabilities["spam"].items()}
        probabilities["ham"] = {word: prob / norm_ham for word, prob in probabilities["ham"].items()}
        return probabilities
    
    def predict(self, text):
        """
        Predict the label of the text by calculating the log of posterior probabilities.
        Parameters:
            text (str): The text to predict
        Returns:
            str: The predicted label
        """
        words = clean_text(text)
        log_prop_spam = np.log(self.phi)
        log_prop_ham = np.log(1 - self.phi)

        for vocab in self.vocab:
            if vocab in words:
                log_prop_spam += np.log(self.cond_props["spam"][vocab] )
                log_prop_ham += np.log(self.cond_props["ham"][vocab] )
            else:
                log_prop_spam += np.log(1 - self.cond_props["spam"][vocab] )
                log_prop_ham += np.log(1 - self.cond_props["ham"][vocab] )


        return "spam" if log_prop_spam > log_prop_ham else "ham"


NB = NaiveBayes(data_train)
y_pred = []
for text in data_valid["text"]:
    y_pred.append(NB.predict(text)) 

print("The accuracy on validating set is using naive bayes: ", np.mean(y_pred == data_valid["label"]))



