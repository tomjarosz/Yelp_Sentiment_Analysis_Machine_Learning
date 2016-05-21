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

labels = ['complaints', 'compliments', 'suggestions for user', 'suggestion for busn']
models_dict = {}

complaint_kwords = list(set(open("data/word_list/complaints.txt").read().splitlines()))
compliments_kwords = list(set(open("data/word_list/compliments.txt").read().splitlines()))
suggestions_busn_kwords = list(set(open("data/word_list/suggestion_busn.txt").read().splitlines()))
suggestions_user_kwords = list(set(open("data/word_list/suggestion_user.txt").read().splitlines()))

models_dict['complaints'] = complaint_kwords
models_dict['compliments'] = complaint_kwords
models_dict['suggestions for user'] = suggestions_user_kwords
models_dict['suggestion for busn'] = suggestions_user_kwords


def read_csv_to_df(filename):
	df = []
	df = pd.read_csv(filename)

	return df

def get_stopwords():
	'''
	Provides a list of stop words.
	There are 387 stopwords in total.
	'''
	with open('data/word_list/stoplists.csv', 'r') as f:
		stopwords = list()
		for line in f:
			stopwords.append(line.strip())
	return stopwords


def train_test(df, y_label):
	if y_label in labels:
		labels.remove(y_label)
	df_dropped_other_labeled = df.drop(labels, axis = 1)	#exclude all labeled columns but label in focus
	train, test = train_test_split(df_dropped_other_labeled, test_size = 0.2, random_state = 0)

	#baseline accuracy
	#print('baseline_accuracy overall: \n {}'.format(df.groupby(y_label).size()))
	#print('baseline_accuracy train: \n {}'.format(train.groupby(y_label).size()))
	print('baseline_accuracy test: \n {}'.format(1 - test[y_label].mean()))
	
	return train, test


def naivebayes_model(train, test, y_label, word_list, stopwords, polarity_inc = True):
	cv = CountVectorizer(vocabulary = word_list, stop_words = stopwords, ngram_range = (1,2), analyzer = 'word')
	train_x_vector = cv.fit_transform(list(train['review'])).toarray()
	test_x_vector = cv.fit_transform(list(test['review'])).toarray()


	if polarity_inc:
		cv.vocabulary_['polarity'] = len(cv.vocabulary_) - 1 # get column names

		# adding polarity as feature
		train_polarity = np.asarray(train['blob_polarity'])
		train_polarity = np.reshape(train_polarity, (len(train_polarity), 1))
		train_x_vector = np.append(train_x_vector, train_polarity, axis = 1)
		
		test_polarity = np.asarray(test['blob_polarity'])
		test_polarity = np.reshape(test_polarity, (len(test_polarity), 1))
		test_x_vector = np.append(test_x_vector, test_polarity, axis = 1)
	
	train_y_values = np.asarray(train[y_label])

	clf = GaussianNB()
	clf.fit(train_x_vector, train_y_values)

	# prediction
	test_y_values = np.asarray(test[y_label])

	y_pred = clf.predict(test_x_vector)

	print(metrics.accuracy_score(test_y_values, y_pred))
	

if __name__ == '__main__':

	df = read_csv_to_df("data/labeled_overlap_data.csv")

	stopwords = get_stopwords()

	for key in models_dict:
		print(key)
		train, test = train_test(df, key)
		naivebayes_model(train, test, key, models_dict[key], stopwords, False)
