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
    '''
    Performs cross validation on the classifier, X and validates on y_true.
    Generates a dictionary for accuracy_baseline, accuracy, precision, area under the precision-recall curve, and runtime.
    '''
    k = 10
    evaluation_dict = dict()
    start_time = time.time()
    y_predicted = cross_val_predict(clf, X, y_true, cv=k)
    end_time = time.time()

    evaluation_dict["accuracy_baseline"] = 1 - y_true.mean()
    evaluation_dict["accuracy"] = metrics.accuracy_score(y_true, y_predicted)
    evaluation_dict["precision"] = metrics.precision_score(y_true, y_predicted)
    evaluation_dict["roc_auc"] = metrics.roc_auc_score(y_true, y_predicted)
    evaluation_dict["runtime"] = end_time - start_time
    return evaluation_dict