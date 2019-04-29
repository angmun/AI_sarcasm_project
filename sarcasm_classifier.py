# Author: Angelica Munyao and Seongwon Im
# Purpose: Determine Multinomial Naive Bayes classifier accuracy on sarcasm text data
# Citations:
# Reading a JSON file:
# https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#json

# Utilizing a pandas DataFrame object:
# https://pandas.pydata.org/pandas-docs/stable/getting_started/dsintro.html#dataframe
# https://www.geeksforgeeks.org/python-pandas-dataframe/

# Converting pandas Series to numpy array:
# https://pandas.pydata.org/pandas-docs/version/0.18/generated/pandas.Series.values.html

# Working with text data and extracting features:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

# Utilizing sklearn's TfidfVectorizer:
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.CountVectorizer

# Splitting the data accounting for number of examples per class:
# https://scikit-learn.org/stable/modules/cross_validation.html#stratified-shuffle-split

# Multi-class text classification:
# https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f

# Multinomial Naive Bayes Classifier:
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

# Linear Support Vector Classifier:
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

# Tensorflow text classification:
# https://developers.google.com/machine-learning/guides/text-classification/step-3

# Classifier and data processing packages
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from tensorflow.python.keras import layers
from tensorflow.python.keras import Sequential
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
import numpy as np

# Data visualization package
from matplotlib import pyplot as plotr

# Data file reading and processing packages
import pandas
import sklearn_pandas as skp

# First load the data-set so we have access to its contents.
# It is a json file so we open the file and read its contents line by line.
# This results in a pandas DataFrame object.
raw_sarcasm_data_json = pandas.read_json("sarcasm_detection\Sarcasm_Headlines_Dataset.json", lines=True)

# The columns of interest to us are the second and third one, with the article titles and sarcasm classification.
# We get pandas Series objects as a result of the column selection by label.
sarcasm_data = raw_sarcasm_data_json['headline']
sarcasm_target = raw_sarcasm_data_json['is_sarcastic']

# In order to process the text data, we need to split the article headlines into words that will count as our features.
# Covert the Series objects into numpy arrays so we can work on them using sklearn.
sarcasm_data_numpy = sarcasm_data.values
sarcasm_target_numpy = sarcasm_target.values

# Bags of words representation for text feature extraction using TfidfVectorizer.
# Combines the result of using CountVectorizer and TfidfTransformer on the headlines.
count_vect = TfidfVectorizer()

# Results in a sparse document-term matrix (words/unigrams are features).
sarcasm_feature_matrix = count_vect.fit_transform(sarcasm_data_numpy)

# Split the data for training using StratifiedShuffleSplit to balance samples from each class.
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

# Resulting data split indices.
train_ind, test_ind = next(splitter.split(sarcasm_feature_matrix, sarcasm_target_numpy))

# First classifier: Multinomial Naive Bayes (default hyperparameters)
multiNB = MultinomialNB()

# Train the classifier using the training data.
multiNB.fit(sarcasm_feature_matrix[train_ind], sarcasm_target_numpy[train_ind])

# Test out the classifier on testing data.
predicted_sarcasm = multiNB.predict(sarcasm_feature_matrix[test_ind])

# Check the score for predicted sarcasm values.
print(multiNB.score(sarcasm_feature_matrix[test_ind], sarcasm_target_numpy[test_ind]))

# Second classifier: Support Vector Machines (default hyperparameters)
lineSVC = LinearSVC()

# Train the classifier using the training data.
lineSVC.fit(sarcasm_feature_matrix[train_ind], sarcasm_target_numpy[train_ind])

# Test out the classifier on testing data.
predicted_sarcasm = lineSVC.predict(sarcasm_feature_matrix[test_ind])

# Check the score for predicted sarcasm values.
print(lineSVC.score(sarcasm_feature_matrix[test_ind], sarcasm_target_numpy[test_ind]))

# NB: Linear SVM has a better score than Multinomial NB using their respective default hyperparameter values.

# Third classifier: Neural Network
# N-gram vectors:
# Remove rare words from the generated vocabulary to improve the learning process.
# We shall go along with the suggested cap of 20000 (from tensorflow's tutorial) top features to work with.
feat_cap = 20000
feature_selector = SelectKBest(k=min(feat_cap, sarcasm_feature_matrix.shape[1]))
best_sarcasm_features = feature_selector.fit(sarcasm_feature_matrix, sarcasm_target_numpy)

# Sequence vectors:
# Tokenize words from original headline array and create a vocabulary of 20000 of the most frequent features.
token_creator = text.Tokenizer(num_words=feat_cap)
token_creator.fit_on_texts(sarcasm_data_numpy)

# Convert tokens into sequence vectors.
# Results in a list of sequence vectors.
sarcasm_feature_sequence = token_creator.texts_to_sequences(sarcasm_data_numpy)

# We shall go along with the suggested cap of 500 (from tensorflow's tutorial) for sequence length.
seq_cap = 500

max_length = len(max(sarcasm_feature_sequence, key=len))
if max_length > seq_cap:
    max_length = seq_cap

# Fix the sequence length to the cap.
sarcasm_feature_sequence = sequence.pad_sequences(sarcasm_feature_sequence, maxlen=max_length)

# Build the model depending on the means of data preparation (n-gram vectors or sequence vectors)
# N-gram vector model: Using multi-layer perceptrons
# Build a





