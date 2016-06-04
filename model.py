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
def unlabeled_data(labeled_csv, full_csv):
	labeled = pd.read_csv(labeled_csv, encoding = 'cp1252')
	unlabeled = pd.read_csv(full_csv)

	labeled_reviews = set()

	for val in labeled['review_id']:
		labeled_reviews.add(val)

	for val in labeled_reviews:
		unlabeled = unlabeled[unlabeled.review_id != val]

	unlabeled.to_csv('data/unlabeled.csv')

	return unlabeled


def stemmer(row):
	review = row['review']
	stem_review = word_stemmer(str(review))

	return stem_review


def read_data(inputfile_path, encoding_ind = True):
	if encoding_ind:
		df = pd.read_csv(inputfile_path, encoding='cp1252')
	else:
		df = pd.read_csv(inputfile_path)

	df['stem_review'] = df.apply(lambda row: stemmer(row), axis=1)

	#df.to_csv('data/test.csv')

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


stopwords = get_stopwords()
df_labeled = read_data("data/training_scored.csv")
df_full = read_data("data/training_scored.csv")

#df_full = read_data("data/unlabeled.csv")

#df_full = read_data("data/manual_train.csv", False)

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
	add_features['polarity'] = df['blob_polarity'] + 1 # transforming polarity 
	# values to accommodate models that cannot take negative x-values
	add_features['stars'] = df['stars']
	add_features['length'] = feature_sentence_length(df)
	add_features['ct_per_busn'] = feature_counts(df, 'business_id')
	add_features['ct_per_review'] = feature_counts(df, 'review_id')

	return add_features


def vectorize_X_Y(df_labeled, df_full, y_label, models_dict, stopwords, tfidf=True):
	'''
	df: labeled data
	df_full: full unlabeled data set (currently set to a subset of the full, unlabeled data set)
	'''
	df_labeled_train, df_labeled_hide = train_test_split(df_labeled, test_size = 0.2, random_state = 0)

	vocabulary = stem_lexicon(models_dict, y_label)
	cv = CountVectorizer(stop_words=stopwords, ngram_range=(1,3), analyzer='word', min_df = .005)
	# cv = CountVectorizer(stop_words=stopwords, ngram_range=(1,3), analyzer='word', vocabulary=vocabulary)

	X_train = cv.fit_transform(list(df_labeled_train['stem_review'])).toarray()
	X_hide = cv.transform(list(df_labeled_hide['stem_review'])).toarray()
	X_full = cv.transform(list(df_full['stem_review'])).toarray()

	print("shape before adding lexicons \n X_train:{} X_hide: {} X_full: {}".format(X_train.shape, X_hide.shape, X_full))

	# adding keyword lexicon as features
	#print("length of vocab list: {}".format(len(vocabulary)))
	cv_lex = CountVectorizer(vocabulary=vocabulary)
	X_train_lex = cv_lex.fit_transform(list(df_labeled_train['stem_review'])).toarray()
	X_hide_lex = cv_lex.transform(list(df_labeled_hide['stem_review'])).toarray()
	X_full_lex = cv_lex.transform(list(df_full['stem_review'])).toarray()
	#print("first 5 entries of lexicon matrix: {}".format(X_train_lex[:5]))
	#print("shape of X_Train lexicon matrix: {}".format(X_train_lex.shape))


	# updating indices of sparse matrix to include lexicon features
	for feature in cv_lex.vocabulary_:
		cv.vocabulary_[feature] = cv_lex.vocabulary_[feature]

	# adding 
	print("X_train shape: {}, X_hide shape: {}, X_full shape: {}".format(X_train.shape, X_hide.shape, X_full.shape))
	#print("X_train_lex shape: {}, X_hide_lex shape: {}, X_full_lex shape: {}".format(X_train_lex.shape, X_hide_lex.shape, X_full_lex.shape))

	X_train = np.append(X_train, X_train_lex, axis = 1)
	X_hide = np.append(X_hide, X_hide_lex, axis = 1)
	X_full = np.append(X_full, X_full_lex, axis = 1)


	if tfidf:
		tfidf_transformer = TfidfTransformer()
		X_train = tfidf_transformer.fit_transform(X_train).toarray()
		X_hide = tfidf_transformer.fit_transform(X_hide).toarray()
		X_full = tfidf_transformer.fit_transform(X_full).toarray()

	add_features_labeled_train = create_features(df_labeled_train)
	add_features_labeled_hide = create_features(df_labeled_hide)
	add_features_full = create_features(df_full)

	for feature in add_features_labeled_train:
		cv.vocabulary_[feature] = len(cv.vocabulary_) - 1
		new_feature = np.asarray(add_features_labeled_train[feature])
		new_feature = np.reshape(new_feature, (len(new_feature), 1))
		#print("X_train shape: {} new_feature shape: {}".format(X_train.shape, new_feature.shape))
		X_train = np.append(X_train, new_feature, axis = 1)

	for feature in add_features_labeled_hide:
		new_feature = np.asarray(add_features_labeled_hide[feature])
		new_feature = np.reshape(new_feature, (len(new_feature), 1))
		X_hide = np.append(X_hide, new_feature, axis = 1)

	for feature in add_features_full:
		new_feature = np.asarray(add_features_full[feature])
		new_feature = np.reshape(new_feature, (len(new_feature), 1))
		X_full = np.append(X_full, new_feature, axis = 1)
	
	# get y_values
	Y_train = np.asarray(df_labeled_train[y_label])
	Y_hide = np.asarray(df_labeled_hide[y_label])


	return X_train, Y_train, X_full, X_hide, Y_hide


if __name__ == '__main__':
	for y_label in models_dict:
		print("model for:", y_label)
		X_train, Y_train, X_full, X_hide, Y_hide = vectorize_X_Y(df_labeled, df_full, y_label, models_dict, stopwords, tfidf=True)
		print("shape of X_train: {} \n shape of Y_train: {} \n shape of X_full: {} \n shape of X_hide: {} \n shape of Y_hide: {}".format(
			X_train.shape, Y_train.shape, X_full.shape, X_hide.shape, Y_hide.shape))


	#unlabeled_data("data/training_scored.csv", "data/full_data.csv")
	#print(len(unlabeled_df))


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