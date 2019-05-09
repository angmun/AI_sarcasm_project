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

# Sequential model guide:
# https://keras.io/getting-started/sequential-model-guide/

# Understanding activation functions:
# https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0

# Deciding on number of hidden layers and nodes for a neural network:
# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

# Classifier and data processing packages
#from tensorflow.python.keras.preprocessing import sequence
#from tensorflow.python.keras.preprocessing import text
#from tensorflow.python.keras import layers
#from tensorflow.python.keras import Sequential
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import numpy as np
import scipy as sp

import sys

# Data visualization package
from matplotlib import pyplot as plotr

# Data file reading and processing packages
import pandas
# import sklearn_pandas as skp

# First load the data-set so we have access to its contents.
# It is a json file so we open the file and read its contents line by line.
# This results in a pandas DataFrame object.
raw_sarcasm_data_json = pandas.read_json("sarcasm_detection\Sarcasm_Headlines_Dataset.json", lines=True)

# The columns of interest to us are the second and third one, with the article titles and sarcasm classification.
# We get pandas Series objects as a result of the column selection by label.
sarcasm_data = raw_sarcasm_data_json['headline']
sarcasm_target = raw_sarcasm_data_json['is_sarcastic']

# In order to process the text data, we need to split the article headlines into words that will count as our features.
# Convert the Series objects into numpy arrays so we can work on them using sklearn.
sarcasm_data_numpy = sarcasm_data.values # Numpy type of all headlines
sarcasm_target_numpy = sarcasm_target.values # Numpy type of 0s and 1s

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Subset from the data so that we can see if the code works. This part will be discarded later
# sarcasm_data_numpy = sarcasm_data_numpy[0:5000,]
# sarcasm_target_numpy = sarcasm_target.values[0:5000,]
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Bags of words representation for text feature extraction using TfidfVectorizer.
# Combines the result of using CountVectorizer and TfidfTransformer on the headlines.
count_vect = TfidfVectorizer()
# count_vect = TfidfTransformer(sublinear_tf = True, use_idf = True)
# count_vect = CountVectorizer()

# Results in a sparse document-term matrix (words/unigrams are features).
sarcasm_feature_matrix = count_vect.fit_transform(sarcasm_data_numpy)
print("TfidfVectorizer Process Complete!")
print("The dimension of the transformed headline matrix is : ", sarcasm_feature_matrix.shape)

# Update sarcasm feature matrix using SVD instead of PCA
# svd = TruncatedSVD(n_components = 3000)
# sarcasm_feature_matrix_SVD = svd.fit_transform(sarcasm_feature_matrix)
kBest = SelectKBest(k = 20000)
sarcasm_feature_matrix_kBest = kBest.fit_transform(sarcasm_feature_matrix, sarcasm_target_numpy)

print("Dimension reduction Process Complete!")
print("The dimension of the kBest transformed headline matrix is : ", sarcasm_feature_matrix_kBest.shape)

# Split the data for training using StratifiedShuffleSplit to balance samples from each class.
# splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

# Resulting data split indices.
# train_ind, test_ind = next(splitter.split(sarcasm_feature_matrix, sarcasm_target_numpy))

# First classifier: Multinomial Naive Bayes (default hyperparameters)
# multiNB = MultinomialNB()
#
# Train the classifier using the training data.
# multiNB.fit(sarcasm_feature_matrix[train_ind], sarcasm_target_numpy[train_ind])

# Test out the classifier on testing data.
# predicted_sarcasm_NB = multiNB.predict(sarcasm_feature_matrix[test_ind])
#
# # Check the score for predicted sarcasm values.
# print(multiNB.score(sarcasm_feature_matrix[test_ind], sarcasm_target_numpy[test_ind]))
#
# # Second classifier: Support Vector Machines (default hyperparameters)
# lineSVC = LinearSVC()
#
# # Train the classifier using the training data.
# lineSVC.fit(sarcasm_feature_matrix[train_ind], sarcasm_target_numpy[train_ind])
#
# # Test out the classifier on testing data.
# predicted_sarcasm_SVC = lineSVC.predict(sarcasm_feature_matrix[test_ind])
#
# # Check the score for predicted sarcasm values.
# print(lineSVC.score(sarcasm_feature_matrix[test_ind], sarcasm_target_numpy[test_ind]))

# NB: Linear SVM has a better score than Multinomial NB using their respective default hyperparameter values.
#---------------------------------------------------------------------------------------------------------------#
# Pipeline
# 1. Dataset
# 2. Split the dataset to train and test
# 3. CountVectorizer (Converts Headlines -> A Matrix with a bunch of features(words))
# 4. tf-idf(Term Frequency times Inverse Document Frequency) <- Do we need this for these short headlines?
# 5. Feature Scaling / Principal Component Analysis -> Tried TruncatedSVD but sign problem occurred
#                                                      so replaced with KBestSelection
# 6. Classifier (Multinomial Naive Bayes, Linear SVC, Decision Tree, KNN) with different parameters

# Lists to store the values
classifiers = ["MultinomialNB", "SVC", "DecisionTree", "KNN"]
meanScores = list()
stdScores = list()

# Naive Bayes Classifier ---------------------------------------------------------------------------------------
# Pipeline
nbPipe = Pipeline([
    # Parameters
    # n_components: Desired dimensionality of output data | default = 2
    # algorithm: SVD solver to use | default = randomized
    # n_iter: Number of iterations for randomized SVD solver | default = 5

    ('classify', MultinomialNB())
])

# Hyper parameters for Naive Bayes
nbSettings = {
    # 'TruncatedSVD__n_components': [20000],
    'classify__alpha': [0.5, 0.75, 1.0, 1.25, 1.5], # Default = 1.0, 0 means no smoothing
    'classify__fit_prior': [True, False],
    # Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
    # Default = True
    # 'classify__class_prior': [None] # Default = None
}

# Use GridSearchCV function to assess the classifier
nbGrid = GridSearchCV(nbPipe, nbSettings, cv = 5)
nbScores = cross_val_score(nbGrid, sarcasm_feature_matrix_kBest, sarcasm_target_numpy, cv = 5)

print("Multinomial Naive Bayes accuracy mean : ", nbScores.mean())
print("Multinomial Naive Bayes accuracy stdev : ", nbScores.std())
meanScores.append(nbScores.mean())
stdScores.append(nbScores.std())

# SVC Classifier ---------------------------------------------------------------------------------------
# Pipeline
svcPipe = Pipeline([
    ('classify', SVC()),
])

# Hyper parameters for SVC
svcSettings = {
    'classify__C': [0.1, 1, 10], # Penalty parameter C of the error term.
    'classify__kernel':['linear'], # Kernel type to be used in an algorithm / default: 'rbf'
    'classify__gamma': ['auto']
}

# Use GridSearchCV function to assess the classifier
svcGrid = GridSearchCV(svcPipe, svcSettings, cv = 5)
svcScores = cross_val_score(svcGrid, sarcasm_feature_matrix_kBest, sarcasm_target_numpy, cv = 5)

print("Support Vector Machine accuracy mean : ", svcScores.mean())
print("Support Vector Machine accuracy stdev : ", svcScores.std())
meanScores.append(svcScores.mean())
stdScores.append(svcScores.std())

# Decision Tree Classifier ----------------------------------------------------------------------------------
treePipe = Pipeline([
    ('classify', DecisionTreeClassifier()),
])

treeSettings = {
    'classify__min_samples_leaf': [10, 50, 100]
}

treeGrid = GridSearchCV(treePipe, treeSettings, cv = 5)
treeScores = cross_val_score(treeGrid, sarcasm_feature_matrix_kBest, sarcasm_target_numpy, cv = 5)

print("Decision Tree accuracy mean : ", treeScores.mean())
print("Decision Tree accuracy stdev : ", treeScores.std())
meanScores.append(treeScores.mean())
stdScores.append(treeScores.std())

# KNN ----------------------------------------------------------------------------------
knnPipe = Pipeline([
    ('classify', KNeighborsClassifier()),
])

knnSettings = {
    'classify__n_neighbors': [1, 5, 10],
    'classify__weights': ['uniform', 'distance']
}

knnGrid = GridSearchCV(knnPipe, knnSettings, cv = 5)
knnScores = cross_val_score(knnGrid, sarcasm_feature_matrix_kBest, sarcasm_target_numpy, cv = 5)

print("K-nearest neighbor accuracy mean : ", knnScores.mean())
print("K-nearest neighbor accuracy stdev : ", knnScores.std())
meanScores.append(knnScores.mean())
stdScores.append(knnScores.std())

# Plotting
plt.bar(classifiers, meanScores, yerr= stdScores)
plt.title("Classifiers Comparison Plot")
plt.xlabel("Classifiers")
plt.ylabel("Mean Scores")
plt.xticks(classifiers)

plt.show()
plt.clf()

sys.exit()
#---------------------------------------------------------------------------------------------------------------#
# Third classifier: Neural Network
# N-gram vectors:
# Remove rare words from the generated vocabulary to improve the learning process.
# We shall go along with the suggested cap of 20000 (from tensorflow's tutorial) top features to work with.
feat_cap = 20000
feature_selector = SelectKBest(k=min(feat_cap, sarcasm_feature_matrix.shape[1]))

# Results in a scipy sparse matrix.
best_sarcasm_features = feature_selector.fit_transform(sarcasm_feature_matrix, sarcasm_target_numpy)

# Convert it to a numpy array.
best_sarcasm_features_numpy = best_sarcasm_features.toarray()

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
# N-gram vector model: Using multi-layer perceptrons.
# Build a multi-layer network.
the_layers = [layers.Dropout(0.2, input_shape=best_sarcasm_features_numpy.shape),
                layers.Dense(128, activation='sigmoid'),
                layers.Dropout(0.2),
              layers.Dense(128, activation='sigmoid'),
              layers.Dense(1, activation='sigmoid')]
networkMLP = Sequential(layers=the_layers)

# Compile the network.
networkMLP.compile(optimizer='adam')

# Train the model using the data.
training = networkMLP.fit(best_sarcasm_features_numpy[train_ind], sarcasm_target_numpy[train_ind],
                          validation_data=(best_sarcasm_features_numpy[test_ind], sarcasm_target_numpy[test_ind]),
                          )




