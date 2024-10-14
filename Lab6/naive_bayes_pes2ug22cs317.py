import numpy as np
import warnings
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

warnings.filterwarnings("ignore", category=RuntimeWarning)


class NaiveBayesClassifier:
    """
    A simple implementation of the Naive Bayes Classifier for text classification.
    """
    @staticmethod
    def preprocess(sentences, categories):
        """
        Preprocess the dataset to remove stop words, and missing or incorrect labels.

        Args:
            sentences (list): List of sentences to be processed.
            categories (list): List of corresponding labels.

        Returns:
            tuple: A tuple of two lists - (cleaned_sentences, cleaned_categories).
        """
        # TO DO 
        
        cleaned_sentences = []
        cleaned_categories = []

        for sentence, category in zip(sentences, categories):
            if category is None or category == "wrong_label":
                continue  # Skip sentences with missing or incorrect labels

            words = sentence.lower().split()  # Convert sentence to lowercase and split into words
            filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
            cleaned_sentences.append(' '.join(filtered_words))
            cleaned_categories.append(category)

        return cleaned_sentences, cleaned_categories

        
    @staticmethod
    def fit(X, y):
        """
        Trains the Naive Bayes Classifier using the provided training data.
        
        Args:
            X (numpy.ndarray): The training data matrix where each row represents a document
                              and each column represents the presence (1) or absence (0) of a word.
            y (numpy.ndarray): The corresponding labels for the training documents.

        Returns:
            tuple: A tuple containing two dictionaries:
                - class_probs (dict): Prior probabilities of each class in the training set.
                - word_probs (dict): Conditional probabilities of words given each class.
        """
        
        class_probs = {}
        word_probs = {}

        classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)

        # Calculate prior probabilities P(C)
        for class_label, class_count in zip(classes, class_counts):
            class_probs[class_label] = class_count / total_samples

        # Calculate conditional probabilities P(W|C) with Laplace smoothing
        num_classes = len(classes)
        vocab_size = X.shape[1]
        word_counts = {class_label: np.zeros(vocab_size) for class_label in classes}

        for i, class_label in enumerate(y):
            if class_label in classes:
                word_counts[class_label] += X[i]

        for class_label in classes:
            total_words_in_class = np.sum(word_counts[class_label])
            word_probs[class_label] = (word_counts[class_label] + 1) / (total_words_in_class + vocab_size)

        return class_probs, word_probs


    @staticmethod
    def predict(X, class_probs, word_probs, classes):
        """
        Predicts the classes for the given test data using the trained classifier.

        Args:
            X (numpy.ndarray): The test data matrix where each row represents a document
                              and each column represents the presence (1) or absence (0) of a word.
            class_probs (dict): Prior probabilities of each class obtained from the training phase.
            word_probs (dict): Conditional probabilities of words given each class obtained from training.
            classes (numpy.ndarray): The unique classes in the dataset.

        Returns:
            list: A list of predicted class labels for the test documents.
        """
        predictions = []

        for doc in X:
            log_probs = {}
            for class_label in classes:
                log_prob = np.log(class_probs[class_label])  # Start with the log of the prior

                # Add log of the conditional probabilities for each word
                for i, word_present in enumerate(doc):
                    if word_present == 1:
                        log_prob += np.log(word_probs[class_label][i])
                    else:
                        log_prob += np.log(1 - word_probs[class_label][i])

                log_probs[class_label] = log_prob
            # Predict the class with the highest log probability
            predicted_class = max(log_probs, key=log_probs.get)
            predictions.append(predicted_class)
            
        return predictions
    
