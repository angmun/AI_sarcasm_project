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
# https://developers.google.com/machine-learning/guides/text-classification

# Sequential model guide:
# https://keras.io/getting-started/sequential-model-guide/

# Understanding activation functions:
# https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0

# Deciding on number of hidden layers and nodes for a neural network:
# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

# Stacked bar plot
# https://matplotlib.org/gallery/lines_bars_and_markers/bar_stacked.html#sphx-glr-gallery-lines-bars-and-markers-bar-stacked-py

# Convoluted Neural Network Tutorial:
# https://medium.com/@jon.froiland/convolutional-neural-networks-for-sequence-processing-part-1-420dd9b500

# Classifier and data processing packages
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from tensorflow.python.keras import layers
from tensorflow.python.keras import Sequential
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
# import numpy as np
# import scipy as sp

# import sys

# Data visualization package
from matplotlib import pyplot as plotr

# Data file reading and processing packages
import pandas


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
# count_vect = TfidfTransformer(sublinear_tf = True, use_idf = True)
# count_vect = CountVectorizer()
count_vect = TfidfVectorizer(ngram_range=(1, 2))

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

print("First Level Complete!")
# #---------------------------------------------------------------------------------------------------------------#

# Second classifier group: Neural Networks
# N-gram Vector Model: Using Multi-Layer Perceptron Neural Networks.
# N-gram vectors:
# Remove rare words from the generated vocabulary to improve the learning process.
# We shall go along with the suggested cap of 20000 (from tensorflow's tutorial) top features to work with.
feat_cap = 20000
feature_selector = SelectKBest(k=min(feat_cap, sarcasm_feature_matrix.shape[1]))

# Results in a scipy sparse matrix.
best_sarcasm_features = feature_selector.fit_transform(sarcasm_feature_matrix, sarcasm_target_numpy)

# Convert it to a numpy array.
best_sarcasm_features_numpy = best_sarcasm_features.toarray()

# Build the model depending on the means of data preparation (n-gram vectors or sequence vectors)
# N-gram vector model: Using multi-layer perceptrons.
# Split the data for training using StratifiedShuffleSplit to balance samples from each class.
splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=0)
splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

# Resulting data split indices.
train_ind, other_ind = next(splitter1.split(best_sarcasm_features_numpy, sarcasm_target_numpy))
val_ind, accest_ind = next(splitter2.split(best_sarcasm_features_numpy[other_ind], sarcasm_target_numpy[other_ind]))

# Build a multi-layer network.
# 64 input for Dense layers:
# With no dropout layers:
the_layers = [layers.Dense(64, activation='sigmoid'),
              layers.Dense(64, activation='sigmoid'),
              layers.Dense(1, activation='sigmoid')]
networkMLP = Sequential(layers=the_layers)

# Configure the network in preparation for training.
networkMLP.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

# Train the model using the data.
# Results in a history object that contains training and validation loss and metrics values.
# History dictionary keys => ['val_loss', 'val_acc', 'loss', 'acc']
training064 = networkMLP.fit(best_sarcasm_features_numpy[train_ind], sarcasm_target_numpy[train_ind],
                             validation_data=(best_sarcasm_features_numpy[val_ind], sarcasm_target_numpy[val_ind]),
                             epochs=10)

# Get the evaluation accuracy value of the network model using the acc_est data chunk.
# Results in a list object that contains loss and accuracy values.
# Metrics names => ['loss', 'acc']
evaluating064 = networkMLP.evaluate(best_sarcasm_features_numpy[accest_ind], sarcasm_target_numpy[accest_ind])

# With one dropout layer:
the_layers = [layers.Dropout(0.2),
              layers.Dense(64, activation='sigmoid'),
              layers.Dense(64, activation='sigmoid'),
              layers.Dense(1, activation='sigmoid')]
networkMLP = Sequential(layers=the_layers)

# Configure the network in preparation for training.
networkMLP.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

# Train the model using the data.
# Results in a history object that contains training and validation loss and metrics values.
training164 = networkMLP.fit(best_sarcasm_features_numpy[train_ind], sarcasm_target_numpy[train_ind],
                             validation_data=(best_sarcasm_features_numpy[val_ind], sarcasm_target_numpy[val_ind]),
                             epochs=10)

# Get the evaluation accuracy value of the network model using the acc_est data chunk.
# Results in a list object that contains loss and accuracy values.
evaluating164 = networkMLP.evaluate(best_sarcasm_features_numpy[accest_ind], sarcasm_target_numpy[accest_ind])

# With two dropout layers:
the_layers = [layers.Dropout(0.2),
              layers.Dense(64, activation='sigmoid'),
              layers.Dropout(0.2),
              layers.Dense(64, activation='sigmoid'),
              layers.Dense(1, activation='sigmoid')]
networkMLP = Sequential(layers=the_layers)

# Configure the network in preparation for training.
networkMLP.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

# Train the model using the data.
# Results in a history object that contains training and validation loss and metrics values.
training264 = networkMLP.fit(best_sarcasm_features_numpy[train_ind], sarcasm_target_numpy[train_ind],
                             validation_data=(best_sarcasm_features_numpy[val_ind], sarcasm_target_numpy[val_ind]),
                             epochs=10)

# Get the evaluation accuracy value of the network model using the acc_est data chunk.
# Results in a list object that contains loss and accuracy values.
evaluating264 = networkMLP.evaluate(best_sarcasm_features_numpy[accest_ind], sarcasm_target_numpy[accest_ind])

# 128 input for Dense layers:
# With no dropout layers:
the_layers = [layers.Dense(128, activation='sigmoid'),
              layers.Dense(128, activation='sigmoid'),
              layers.Dense(1, activation='sigmoid')]
networkMLP = Sequential(layers=the_layers)

# Configure the network in preparation for training.
networkMLP.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

# Train the model using the data.
# Results in a history object that contains training and validation loss and metrics values.
training0128 = networkMLP.fit(best_sarcasm_features_numpy[train_ind], sarcasm_target_numpy[train_ind],
                              validation_data=(best_sarcasm_features_numpy[val_ind], sarcasm_target_numpy[val_ind]),
                              epochs=10)

# Get the evaluation accuracy value of the network model using the acc_est data chunk.
# Results in a list object that contains loss and accuracy values.
evaluating0128 = networkMLP.evaluate(best_sarcasm_features_numpy[accest_ind], sarcasm_target_numpy[accest_ind])

# With one dropout layer:
the_layers = [layers.Dropout(0.2),
              layers.Dense(128, activation='sigmoid'),
              layers.Dense(128, activation='sigmoid'),
              layers.Dense(1, activation='sigmoid')]
networkMLP = Sequential(layers=the_layers)

# Configure the network in preparation for training.
networkMLP.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

# Train the model using the data.
# Results in a history object that contains training and validation loss and metrics values.
training1128 = networkMLP.fit(best_sarcasm_features_numpy[train_ind], sarcasm_target_numpy[train_ind],
                              validation_data=(best_sarcasm_features_numpy[val_ind], sarcasm_target_numpy[val_ind]),
                              epochs=10)

# Get the evaluation accuracy value of the network model using the acc_est data chunk.
# Results in a list object that contains loss and accuracy values.
evaluating1128 = networkMLP.evaluate(best_sarcasm_features_numpy[accest_ind], sarcasm_target_numpy[accest_ind])

# With two dropout layers:
the_layers = [layers.Dropout(0.2),
              layers.Dense(128, activation='sigmoid'),
              layers.Dropout(0.2),
              layers.Dense(128, activation='sigmoid'),
              layers.Dense(1, activation='sigmoid')]
networkMLP = Sequential(layers=the_layers)

# Configure the network in preparation for training.
networkMLP.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

# Train the model using the data.
# Results in a history object that contains training and validation loss and metrics values.
training2128 = networkMLP.fit(best_sarcasm_features_numpy[train_ind], sarcasm_target_numpy[train_ind],
                              validation_data=(best_sarcasm_features_numpy[val_ind], sarcasm_target_numpy[val_ind]),
                              epochs=10)

# Get the evaluation accuracy value of the network model using the acc_est data chunk.
# Results in a list object that contains loss and accuracy values.
evaluating2128 = networkMLP.evaluate(best_sarcasm_features_numpy[accest_ind], sarcasm_target_numpy[accest_ind])

# Plot the learning curves for the six configurations of the network:
plotr.plot(training064.epoch, training064.history['val_acc'], label='64 units, 0 dropout')
plotr.plot(training164.epoch, training164.history['val_acc'], label='64 units, 1 dropout')
plotr.plot(training264.epoch, training264.history['val_acc'], label='64 units, 2 dropouts')
plotr.plot(training0128.epoch, training0128.history['val_acc'], label='128 units, 0 dropout')
plotr.plot(training1128.epoch, training1128.history['val_acc'], label='128 units, 1 dropout')
plotr.plot(training2128.epoch, training2128.history['val_acc'], label='128 units, 2 dropouts')
plotr.title('Sarcasm detection learning curve')
plotr.xlabel("Training epochs")
plotr.ylabel("Network Accuracy")
plotr.legend()
plotr.show()
plotr.clf()

# Plot the bar graph for accuracy values of the network configurations when evaluated using the acc_est data split:
# The bar positions on the graph:
bar_pos = [x for x in range(6)]
model_config = ['64 Units\n0 Dropouts', '64 Units\n1 Dropouts', '64 Units\n2 Dropouts', '128 Units\n0 Dropouts',
                '128 Units\n1 Dropouts', '128 Units\n2 Dropouts']
model_loss = [evaluating064[0], evaluating164[0], evaluating264[0], evaluating0128[0], evaluating1128[0],
              evaluating2128[0]]
model_acc = [evaluating064[1], evaluating164[1], evaluating264[1], evaluating0128[1], evaluating1128[1],
             evaluating2128[1]]
p1 = plotr.bar(bar_pos, model_acc, 0.5)
p2 = plotr.bar(bar_pos, model_loss, 0.5, bottom=model_acc)
plotr.ylabel('Model Accuracy')
plotr.xlabel('Model Configuration')
plotr.xticks(bar_pos, model_config)
plotr.title('Sarcasm detection MLP model accuracy')
plotr.legend((p1[0], p2[0]), ('Accuracy', 'Loss'))
plotr.show()
plotr.clf()

# Sequence vector model: Using convoluted neural network.
# Sequence vectors:
# Tokenize words from original headline array and create a vocabulary of 20000 of the most frequent features.
token_creator = text.Tokenizer(num_words=feat_cap)
token_creator.fit_on_texts(sarcasm_data_numpy)

# The length of the generated word dictionary for future use.
dict_len = len(token_creator.word_index)

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

# Save the length of a sequence for future use.
seq_length = len(sarcasm_feature_sequence[0])

# Split the data for training using StratifiedShuffleSplit to balance samples from each class.
splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=0)
splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)

# Resulting data split indices.
train_ind, other_ind = next(splitter1.split(sarcasm_feature_sequence, sarcasm_target_numpy))
val_ind, accest_ind = next(splitter2.split(sarcasm_feature_sequence[other_ind], sarcasm_target_numpy[other_ind]))
the_cnn_layers = [layers.Embedding(dict_len, 64, input_length=seq_length),
                  layers.Dropout(0.2),
                  layers.SeparableConv1D(32, 5, activation='sigmoid', bias_initializer='random_uniform',
                                         depthwise_initializer='random_uniform',
                                         padding='same'),
                  layers.MaxPooling1D(5),
                  layers.SeparableConv1D(32, 5, activation='sigmoid', bias_initializer='random_uniform',
                                         depthwise_initializer='random_uniform',
                                         padding='same'),
                  layers.GlobalMaxPooling1D(),
                  layers.Dense(1, activation='sigmoid')
                  ]
networkCNN = Sequential(layers=the_cnn_layers)

# Configure the network in preparation for training.
networkCNN.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

# Train the model using the data.
# Results in a history object that contains training and validation loss and metrics values.
trainingCNN = networkCNN.fit(sarcasm_feature_sequence[train_ind], sarcasm_target_numpy[train_ind],
                             validation_data=(sarcasm_feature_sequence[val_ind], sarcasm_target_numpy[val_ind]),
                             epochs=10)

# Get the evaluation accuracy value of the network model using the acc_est data chunk.
# Results in a list object that contains loss and accuracy values.
evaluatingCNN = networkCNN.evaluate(sarcasm_feature_sequence[accest_ind], sarcasm_target_numpy[accest_ind])

# Plot the learning curve for the convoluted neural network.
plotr.plot(trainingCNN.epoch, trainingCNN.history['val_acc'], label='Val_Acc')
plotr.plot(trainingCNN.epoch, trainingCNN.history['val_loss'], label='Val_Loss')
plotr.title('Sarcasm detection learning curve sepCNN')
plotr.xlabel("Training epochs")
plotr.ylabel("Network Metric Values")
plotr.legend()
plotr.show()
plotr.clf()
print("Accuracy: {}, Loss: {}".format(evaluatingCNN[1], evaluatingCNN[0]))
