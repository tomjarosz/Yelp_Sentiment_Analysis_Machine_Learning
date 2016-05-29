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
from sklearn.feature_extraction.text import TfidfTransformer


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

# Preprocessing
def read_data(inputfile_path, encoding_ind = True):
	if encoding_ind:
		df = pd.read_csv(inputfile_path, encoding='cp1252')
	else:
		df = pd.read_csv(inputfile_path)

	df['stem_review'] = df.apply(lambda row: stemmer(row), axis=1)

	#df.to_csv('data/test.csv')

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


stopwords = get_stopwords()
df_train = read_data("data/training_scored.csv")
df_full = read_data("data/training_data.csv", False)

# Feature Creation
def stem_lexicon(models_dict, key):
	word_list_stem = []
	for val in models_dict[key]:
		stem = word_stemmer(val)
		if stem not in word_list_stem:
			word_list_stem.append(stem)
	
	return word_list_stem


#def feature_preced_label(df):
#	preceding_label = []

#	for i in df['stem_review'].loc[row_indexer]:
#		length.append(len(val))

#	return length 

def feature_sentence_length(df):
	length = []

	for val in df['stem_review']:
		length.append(len(val))

	return length 


def feature_counts(df, category):

	ct_per_cat = df_full.groupby(category).size()
	l_ct_per_cat = []

	for cat_id in df[category]:
		count = ct_per_cat[cat_id]
		l_ct_per_cat.append(count)

	return l_ct_per_cat


def create_features(df):
	add_features = {}
	add_features['polarity'] = df['blob_polarity']
	add_features['stars'] = df['stars']
	add_features['length'] = feature_sentence_length(df)
	add_features['ct_per_busn'] = feature_counts(df, 'business_id')
	add_features['ct_per_review'] = feature_counts(df, 'review_id')

	return add_features


def vectorize_X_Y(df_train, df_full, y_label, models_dict, stopwords, tfidf=True):
	'''
	df: labeled data
	df_full: full unlabeled data set (currently set to a subset of the full, unlabeled data set)
	'''

	vocabulary = stem_lexicon(models_dict, y_label)
	cv = CountVectorizer(stop_words=stopwords, ngram_range=(1,3), analyzer='word', vocabulary = vocabulary)
	X = cv.fit_transform(list(df_train['stem_review'])).toarray()
	X_full = cv.fit_transform(list(df_full['stem_review'])).toarray()

	if tfidf:
		tfidf_transformer = TfidfTransformer()
		X = tfidf_transformer.fit_transform(X).toarray()
		X_full = tfidf_transformer.fit_transform(X_full).toarray()

	add_features_train = create_features(df_train)
	add_features_full = create_features(df_full)

	for feature in add_features_train:
		cv.vocabulary_[feature] = len(cv.vocabulary_) - 1
		new_feature = np.asarray(add_features_train[feature])
		new_feature = np.reshape(new_feature, (len(new_feature), 1))
		X = np.append(X, new_feature, axis = 1)

	for feature in add_features_full:
		new_feature = np.asarray(add_features_full[feature])
		new_feature = np.reshape(new_feature, (len(new_feature), 1))
		X_full = np.append(X_full, new_feature, axis = 1)
	
	# get y_values
	Y = np.asarray(df[y_label])

	return X, Y, X_full


if __name__ == '__main__':
	for y_label in models_dict:
		print("model for:", y_label)
		X, Y, X_full = vectorize_X_Y(df, df_full, y_label, models_dict, stopwords, tfidf=True)
		print("shape of X: {} \n", "shape of Y: {} \n", "shape of X_full: {} \n",
			X.shape, Y.shape, X_full.shape)



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

	if polarity_inc: # adding polarity as feature
		cv.vocabulary_['polarity'] = len(cv.vocabulary_) - 1 # get column names
		new_feature = np.asarray(df['blob_polarity'])
		new_feature = np.reshape(new_feature, (len(new_feature), 1))
		X = np.append(X, new_feature, axis = 1)

	if star_inc: # adding polarity as feature
		cv.vocabulary_['stars'] = len(cv.vocabulary_) - 1 # get column names
		new_feature = np.asarray(df['stars'])
		new_feature = np.reshape(new_feature, (len(new_feature), 1))
		X = np.append(X, new_feature, axis = 1)

'''