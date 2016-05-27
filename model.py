# Draft of model to classify complaints, compliments, suggestions for public 
# institutions listed in Yelp


## prelim results:
	# test data: 26 labeled complaints out of 102 (74.5098% accuracy if guessing)
	# when passing just polarity: accuracy score is 74.5098%
	# when passing just the complaint_words: accurayc score is 75.4902%
		# .9% increase in a accuracy!
	# stemmed words and features
	# don't think polarity matters
	# compliments.txt not helpful
	# tom's models won on complaints and suggestions (even without the keywords)
	# sirui's model won on compliments (without the keywords)

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pylab as plt
from sklearn import (preprocessing, cross_validation, svm, metrics, tree, 
    decomposition, svm)
import csv
from word_stemmer import word_stemmer
from preprocess import *

clf = [	GaussianNB(), MultinomialNB(), BernoulliNB() ]

labels = ['complaint', 'compliment', 'suggestion for user', 'suggestion for business']
models_dict = {}

complaint_kwords = list(set(open("data/word_list/complaints.txt").read().splitlines()))
compliments_kwords = list(set(open("data/word_list/compliments.txt").read().splitlines()))
suggestions_busn_kwords = list(set(open("data/word_list/suggestion_busn.txt").read().splitlines()))
suggestions_user_kwords = list(set(open("data/word_list/suggestion_user.txt").read().splitlines()))

models_dict['complaint'] = complaint_kwords
models_dict['compliments'] = compliments_kwords
models_dict['suggestion for user'] = suggestions_user_kwords
models_dict['suggestion for business'] = suggestions_busn_kwords

stopwords = get_stopwords()

df = read_data("data/training_scored.csv")

def stem_lexicon(models_dict, key):
	word_list_stem = []
	for val in models_dict[key]:
		stem = word_stemmer(val)
		if stem not in word_list_stem:
			word_list_stem.append(stem)
	
	return word_list_stem


def vectorize_X_Y(df, y_label, models_dict, stopwords, tfidf=True, polarity_inc=True):

	vocabulary = stem_lexicon(models_dict, y_label)
	cv = CountVectorizer(stop_words=stopwords, ngram_range=(1,2), analyzer='word', vocabulary = vocabulary)
	X = cv.fit_transform(list(df['stem_review'])).toarray()

	if tfidf:
		tfidf_transformer = TfidfTransformer()
		X = tfidf_transformer.fit_transform(X).toarray()

	if polarity_inc: # adding polarity as feature
		cv.vocabulary_['polarity'] = len(cv.vocabulary_) - 1 # get column names
		polarity = np.asarray(df['blob_polarity'])
		print("length:", len(polarity))
		polarity = np.reshape(polarity, (len(polarity), 1))
		X = np.append(X, polarity, axis = 1)
	
	# get y_values
	Y = np.asarray(df[y_label])

	return X, Y


if __name__ == '__main__':
	for y_label in models_dict:
		print("model for:", y_label)
		X, Y = vectorize_X_Y(df, y_label, models_dict, stopwords, tfidf=True, polarity_inc=True)


''' ARCHIVE:: 
def naivebayes_model(train, test, y_label, word_list, stopwords, polarity_inc = True):
	cv = CountVectorizer(vocabulary = word_list, stop_words = stopwords, ngram_range = (1,2), analyzer = 'word')
	train_x_vector = cv.fit_transform(list(train['stem_review'])).toarray()
	test_x_vector = cv.fit_transform(list(test['stem_review'])).toarray()

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



	clf.fit(train_x_vector, train_y_values)

	# prediction
	test_y_values = np.asarray(test[y_label])

	y_pred = clf.predict(test_x_vector)

	print(metrics.accuracy_score(test_y_values, y_pred))
'''