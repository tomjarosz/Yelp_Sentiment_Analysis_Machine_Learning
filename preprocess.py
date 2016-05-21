# Author: Sirui Feng
# This file vectorizes text.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
import csv
import numpy as np
import pandas as pd

inputfile_path = 'data/labeled_overlap_data.csv'

def get_X_and_Y(training_df, testing_df):
	
	X_train = training_df['review']
	X_test = testing_df['review']

	X_train_vector, X_test_vector = vectorize_X(training_df, testing_df)

	labels = ['complaint', 'suggestion for user', 'compliments', 'neutral', 'suggestion for business']
	for label in labels:
		Y_train = training_df[label]

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

def vectorize_X(training_df, testing_df, tfidf=True):
	stopwords = get_stopwords()
	vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1,2), analyzer='word')
		
	X_train_count = vectorizer.fit_transform(training_data['review'])
	X_test_count = vectorizer.transform()

	if tfidf:

		tfidf_transformer = TfidfTransformer()
		X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)
		X_test_tfidf = tfidf_transformer.transform(X_test_count)

		return X_train_tfidf, X_test_tfidf

	return X_train_count, X_test_count

def pca_vectorize(training_data, n_components):
	pca = PCA(n_components = n_components)
	X_new = pca.fit_transform(training_data)
