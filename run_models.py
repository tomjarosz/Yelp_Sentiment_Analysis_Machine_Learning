import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import json
import numpy as np
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.metrics import precision_recall_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from word_stemmer import word_stemmer
from model import *
from cross_validation import cross_validation
from classify import classify

def run_models():

    models = ['SVM', 'SGD', 'NBMultinomial', 'NBGaussian', 'NBBernoulli']

    clfs = {'SVM': svm.LinearSVC(random_state=0, dual=False),
            'SGD': SGDClassifier(loss="hinge", penalty="l2"),
            'NBMultinomial': MultinomialNB(alpha=1, fit_prior=True, class_prior=None),
            'NBGaussian': GaussianNB(),
            'NBBernoulli': BernoulliNB(alpha=1, binarize=0.0, fit_prior=True, class_prior=None)}

    grid = {'SVM' :{'C' :[0.0001,0.01,1,10],'penalty':['l1', 'l2']},
            'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
            'NBMultinomial': {'alpha': [0.0001, 0.01, 1, 2]},
            'NBGaussian' : {},
            'NBBernoulli': {'alpha': [0.0001, 0.01, 1, 2]}}


    labels = ['complaint', 'compliments', 'suggestion for user', 'suggestion for business']
    for label in labels:
        x_train, y_train, x_full, x_hide, y_hide = vectorize_X_Y(df_labeled, df_full, label, models_dict, stopwords, tfidf=True)

        label_dict = {}

        for index,clf in enumerate([clfs[x] for x in models]):
            parameter_values = grid[models[index]]
            for p in ParameterGrid(parameter_values):
                clf.set_params(**p)

                label_dict[clf] = cross_validation(clf, x_train, y_train)

        print(label_dict)
                
        best_clf = None
        best_precision = 0
        for classifier in label_dict:
            current_precision = label_dict[classifier]['precision']
            if current_precision > best_precision:
                best_clf = classifier
                best_precision = current_precision
       
        y_full_predict = classify(best_clf, x_train, y_train, x_full)
        y_full_predict = np.reshape(y_full_predict, (len(y_full_predict), 1))
        x_full = np.append(x_full, y_full_predict, axis = 1)
        # np.savetxt('x_full' + str(label) + '.csv', x_full, delimiter = ",")
        # print("x_full shape:", x_full.shape)

  
if __name__ == '__main__':
    run_models()