# Author: Sirui Feng
# This file preprocesses the data and performs k-fold cross validation.

'''

'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import metrics
from word_stemmer import word_stemmer
import csv
import numpy as np
import pandas as pd
import time
from sklearn import cross_validation
from sklearn.cross_validation import KFold

inputfile_path = 'data/training_scored.csv'

labels = ['complaint', 'compliments', 'suggestion for user', 'suggestion for business']
models_dict = {}

complaint_kwords = list(set(open("data/word_list/complaints.txt").read().splitlines()))
complaint_kwords_stem=set()
for w in complaint_kwords:
	complaint_kwords_stem.add(word_stemmer(w))

compliments_kwords = list(set(open("data/word_list/compliments.txt").read().splitlines()))
compliments_kwords_stem=set()
for w in compliments_kwords:
	compliments_kwords_stem.add(word_stemmer(w))

suggestions_busn_kwords = list(set(open("data/word_list/suggestion_busn.txt").read().splitlines()))
suggestions_busn_kwords_stem=set()
for w in suggestions_busn_kwords:
	suggestions_busn_kwords_stem.add(word_stemmer(w))

suggestions_user_kwords = list(set(open("data/word_list/suggestion_user.txt").read().splitlines()))
suggestions_user_kwords_stem=set()
for w in suggestions_user_kwords:
	suggestions_user_kwords_stem.add(word_stemmer(w))

models_dict['complaint'] = complaint_kwords_stem
models_dict['compliments'] = complaint_kwords_stem
models_dict['suggestion for user'] = suggestions_user_kwords_stem
models_dict['suggestion for business'] = suggestions_user_kwords_stem

def read_data(inputfile_path):
	df = pd.read_csv(inputfile_path, encoding='cp1252')

	df['stem_review'] = df.apply(lambda row: stemmer(row), axis=1)

	df.to_csv('data/test.csv')

	return df

def stemmer(row):
	review = row['review']
	stem_review = word_stemmer(review)

	return stem_review
	
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

def split_training_testing(df):
	training_df, testing_df = train_test_split(df, test_size = 0.2, random_state = 0)
	return training_df, testing_df

# def vectorize_X(X_train, X_test, vocabulary, tfidf=True):
# 	stopwords = get_stopwords()
# 	#vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1,2), analyzer='word', vocabulary = vocabulary)
# 	vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1,2), analyzer='word')
		
# 	X_train = vectorizer.fit_transform(X_train)
# 	X_test = vectorizer.transform(X_test)
		
# 	if tfidf:

# 		tfidf_transformer = TfidfTransformer()
# 		X_train = tfidf_transformer.fit_transform(X_train)
# 		X_test = tfidf_transformer.transform(X_test)

# 	return X_train, X_test

def vectorize_x(X):
	stopwords = get_stopwords()
	#vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1,2), analyzer='word', vocabulary = vocabulary)
	vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1,2), analyzer='word')
	X = vectorizer.fit_transform(X)
	if tfidf:
		tfidf_transformer = TfidfTransformer()
		X = tfidf_transformer.fit_transform(X)
	return X


def get_X_and_Y(training_df, testing_df, label):
	
	X_train = training_df['stem_review']

	X_test = testing_df['stem_review']

	vocabulary = models_dict[label]

	X_train, X_test = vectorize_X(training_df, testing_df, vocabulary)
	print("~~~~~~~~~~~~~~~~~~~~~")
	#print(X_train)
	#print(X_train.shape)

	Y_train = training_df[label]
	Y_true = testing_df[label]

	#print(np.any(np.isnan(Y_train)))

	#print(label)
	#print(Y_train.shape)
	#print(X_train.shape)
	print(Y_train.describe())

	return X_train, X_test, Y_train, Y_true

def get_predictions(clf, X_train, y_train, X_test):
	clf.fit(X_train,y_train)
	y_predict = clf.predict(X_test)

	return y_predict

def NaiveBayesClf(X_train, X_test, y_train, y_test):
	print(y_train.describe())
	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	y_predict = get_predictions(clf, X_train, y_train, X_test)
	print('Baseline:', 1-y_test.mean())
	print()
	print(metrics.accuracy_score(y_test, y_predict))


# def NaiveBayesClf(training_df, testing_df):
# 	labels = ['complaint', 'compliments', 'suggestion for user', 'suggestion for business']

# 	#labels = ['complaint', 'suggestion for user', 'compliments', 'neutral', 'suggestion for business']
# 	for label in labels:

# 		X_train, X_test, Y_train, Y_true = get_X_and_Y(training_df, testing_df, label)



# 		clf = MultinomialNB()
# 		clf.fit(X_train,Y_train)
# 		Y_predict = get_predictions(clf, X_train, Y_train, X_test)
# 		print("Baseline:", 1-Y_true.mean())
# 		print()


# 		print(metrics.accuracy_score(Y_true, Y_predict))

def cross_validation(clf, X, y):
	k = 7

	evaludation_dict = dict()

	scores = cross_validation.cross_val_score(clf, X, y_true, cv=k)
	print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	y_predicted = cross_validation.cross_val_predict(clf, X, y, cv=k)

	evaludation_dict["accuracy"] = metrics.accuracy_score(y_true, y_predicted)
	evaludation_dict["precision"] = metrics.precision_score(y_true, y_predicted)
	evaludation_dict["roc_auc"] = metrics.roc_auc_score(y_true, y_predicted)


	return evaludation_dict

def classify(best_clf, X_train, y_train, X_full):
	best_clf.fit(X_train, y_train)
	y_full = get_predictions(best_clf, X_train, Y_train, X_full)
	return y_full

def output_full_to_dict(df_full):

	df_full_predicted = df_full

	return



if __name__ == '__main__':

	df = read_data(inputfile_path)
	print(df.shape)
	#print(df.describe())
	#print(df.shape)
	df_original = df.copy()
	#training_df, testing_df = split_training_testing(df)
	start_time = time.time()
	#NaiveBayesClf(training_df, testing_df)
	X = df['stem_review']
	labels = ['complaint', 'compliments', 'suggestion for user', 'suggestion for business']
	for label in labels:
		y = df[label]
		k_fold = KFold(df.shape[0], n_folds = 6)
		for train_index, test_index in k_fold:
			print("~~~~~~~~~~~~~~~~~")
			#X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
			vocabulary = models_dict[label]
			X_train, X_test = vectorize_X(X_train, X_test, vocabulary)
			NaiveBayesClf(X_train, X_test, y_train, y_test)
	end_time = time.time()
	print("Naive Bayes takes", end_time-start_time, "seconds")

# def pca_vectorize(training_data, n_components):
# 	pca = PCA(n_components = n_components)
# 	X_new = pca.fit_transform(training_data)
