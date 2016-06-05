import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import json
import numpy as np
import csv
import time
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

    grid = {'SVM' :{'C' :[0.0001,0.01,10,1],'penalty':['l1', 'l2']},
            'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
            'NBMultinomial': {'alpha': [0.0001, 0.01, 1, 2]},
            'NBGaussian' : {},
            'NBBernoulli': {'alpha': [ 0.01, 1, 2, 0.0001]}}


    labels = ['complaint', 'compliments', 'suggestion for user', 'suggestion for business']

    with open('performance_report.csv', 'w') as outfile:
        w = csv.writer(outfile, delimiter=',')
        w.writerow(['Label', 'Classifier','Accuracy_Baseline', 'Accuracy', 'Precision', 'AUC', "Runtime"])
        for label in labels:

            best_model = None
            #best_params = None
            best_score = -1

            print(label)
            x_train, y_train, x_full, x_hide, y_hide = vectorize_X_Y(df_labeled, df_full, label, models_dict, stopwords, tfidf=True)

            #label_dict = {}
            

            for index,clf in enumerate([clfs[x] for x in models]):
                
                # best_precision = -1
                # best_perform = None
                # best_param = None
                #current_model = models[index]

                parameter_values= grid[models[index]]
                #temp = dict()

                for p in ParameterGrid(parameter_values):

                    #current_params = p
                    clf.set_params(**p)
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    #current_clf = clf
                    print(clf)
                    #print(current_params)

                    temp = cross_validation(clf, x_train, y_train)
                    print(temp)

                    w.writerow([label, clf, temp["accuracy_baseline"], \
                    temp['accuracy'], temp['roc_auc'], \
                    temp['roc_auc'], temp['runtime']])
                    
                    if temp['roc_auc'] > best_score:
                        #best = clf
                        best_score = temp['roc_auc']
                        #best_perform = temp
                        #best_params = p
                        best_model = clf
                        print("+++++Best model has been changed to:", best_model)
                        #print("+++++Best param has been changed to:", best_params)

            #best_overall_model= clfs[best_model]
            #best = best_overall_model.set(**best_params)
            print("--> my best clf is:", best_model)
                
            print("="*60)
            #print('my final best clf is:', best)
                
                # label_dict[best] = best_perform
                # print(label_dict)
                    
            # for classifier in label_dict:

            #     current_score = label_dict[classifier]['precision']
            #     if current_score > best_overall_score:
            #         best_overall_clf = classifier
            #         best_overall_score= current_score
       
            y_full_predict = classify(best_model, x_train, y_train, x_full)
            df_full[label] = y_full_predict

    df_full.to_csv('result.csv')
  
if __name__ == '__main__':
    run_models()
