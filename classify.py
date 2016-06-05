# Author: Sirui Feng
# This file classifies the full dataset with the best-performed classifier on training set.
# Call classify()

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


def get_predictions(clf, X_train, y_train_true, X_full):
	'''
	Helper function for classify
	'''
	clf.fit(X_train,y_train_true)
	y_full_predict = clf.predict(X_full)

	return y_full_predict

def classify(best_clf, X_train, y_train_true, X_full):
	'''
	Trains the clf on X_train and y_train_true, and predict X_full.
	'''
	best_clf.fit(X_train, y_train_true)
	y_full_predict = get_predictions(best_clf, X_train, y_train_true, X_full)
	return y_full_predict