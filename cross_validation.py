# Author: Sirui Feng
# This file performs cross validation on the testing(labeled) dataset.

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
from sklearn.cross_validation import cross_val_predict

def cross_validation(clf, X, y_true):
	k = 7

	evaludation_dict = dict()

	#scores = cross_validation.cross_val_score(clf, X, y_true, cv=k)
	#print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print(clf)
    y_predicted = cross_val_predict(clf, X, y_true, cv=k)

	evaludation_dict["accuracy"] = metrics.accuracy_score(y_true, y_predicted)
	evaludation_dict["precision"] = metrics.precision_score(y_true, y_predicted)
	evaludation_dict["roc_auc"] = metrics.roc_auc_score(y_true, y_predicted)


	return evaludation_dict