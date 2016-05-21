# Draft of model to classify complaints, compliments, 
# suggestions for public institutions listed in Yelp


## prelim results:
	# test data: 26 labeled complaints out of 102 (74.5098% accuracy if guessing)
	# when passing just polarity: accuracy score is 74.5098%
	# when passing just the complaint_words: accurayc score is 75.4902%
		# .9% increase in a accuracy!

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pylab as plt
from sklearn import (preprocessing, cross_validation, svm, metrics, tree, 
    decomposition, svm)

import csv

labels = ['complaints', 'compliments', 'neutral', 'suggestions for user', 'suggestion for busn']

complaint_kwords = list(set(open("data/word_list/complaints.txt").read().splitlines()))
compliments_kwords = list(set(open("data/word_list/compliments.txt").read().splitlines()))
suggestions_busn_kwords = list(set(open("data/word_list/suggestion_busn.txt").read().splitlines()))
suggestions_user_kwords = list(set(open("data/word_list/suggestion_user.txt").read().splitlines()))


def read_csv_to_df(filename):
	df = []
	df = pd.read_csv(filename)

	return df


def train_test_set(df, y_label):
	if y_label in labels:
		labels.remove(y_label)
	df_dropped_other_labeled = df.drop(labels, axis = 1)	#exclude all labeled columns but label in focus
	train, test = train_test_split(df_dropped_other_labeled, test_size = 0.2, random_state = 0)

	#baseline accuracy
	print('baseline_accuracy overall: \n {}'.format(df.groupby(y_label).size()))
	print('baseline_accuracy train: \n {}'.format(train.groupby(y_label).size()))
	print('baseline_accuracy test: \n {}'.format(test.groupby(y_label).size()))
	
	#return train, test


def naivebayes_model(train, test, y_value):
	cv = CountVectorizer(vocabulary = complaint_kwords)
	array = cv.fit_transform(list(train['review'])).toarray()

	cv.vocabulary_['polarity'] = len(cv.vocabulary_) - 1 # get column names

	# adding polarity as feature
	polarity = np.asarray(train['blob_polarity'])
	polarity = np.reshape(polarity, (len(polarity), 1))

	x_vector = np.append(array, polarity, axis = 1)
	y_values = np.asarray(train[y_value])

	clf = GaussianNB()
	clf.fit(x_vector, y_values)

	# prediction
	test_x_vector = cv.fit_transform(list(test['review'])).toarray()
	polarity = np.asarray(test['blob_polarity'])
	polarity = np.reshape(polarity, (len(polarity), 1))

	test_x_vector = np.append(array, polarity, axis = 1)
	test_y_values = np.asarray(test[y_value])

	y_pred = clf.predict(test_x_vector)

	print(metrics.accuracy_score(y_true, y_pred))
	

if __name__ == '__main__':

	df = read_csv_to_df("data/labeled_overlap_data.csv")
	train, test = train_test_split(df, 'complaints')
	array = naivebayes_model(train)
