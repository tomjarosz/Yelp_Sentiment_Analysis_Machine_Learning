# Author: Sirui Feng
# This file vectorizes text.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import metrics, svm
from word_stemmer import word_stemmer
import csv
import numpy as np
import pandas as pd

inputfile_path = 'data/labeled_overlap_data.csv'

def read_data(inputfile_path):
    df = pd.read_csv(inputfile_path)

    df['stem_review'] = df.apply(lambda row: stemmer(row), axis=1)

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

def vectorize_X(training_df, testing_df, tfidf=True):
    stopwords = get_stopwords()
    vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1,2), analyzer='word')
        
    X_train = vectorizer.fit_transform(training_df['review'])
    X_test = vectorizer.transform(testing_df['review'])

    if tfidf:

        tfidf_transformer = TfidfTransformer()
        X_train = tfidf_transformer.fit_transform(X_train)
        X_test = tfidf_transformer.transform(X_test)

    return X_train, X_test

def get_X_and_Y(training_df, testing_df, label):
    
    X_train = training_df['review']

    X_test = testing_df['review']

    X_train, X_test = vectorize_X(training_df, testing_df)
    print("~~~~~~~~~~~~~~~~~~~~~")
    #print(X_train)
    print(X_train.shape)

    Y_train = training_df[label]
    Y_true = testing_df[label]

    print(np.any(np.isnan(Y_train)))

    #print(label)
    #print(Y_train.shape)
    #print(X_train.shape)
    print(Y_train.describe())

    return X_train, X_test, Y_train, Y_true

def get_predictions(clf, X_train, Y_train, X_test):
    clf.fit(X_train,Y_train)
    Y_predict = clf.predict(X_test)

    return Y_predict

def SVMClf(training_df, testing_df):

    labels = ['complaints', 'suggestions for user', 'compliments', 'neutral', 'suggestion for busn']
    for label in labels:

        X_train, X_test, Y_train, Y_true = get_X_and_Y(training_df, testing_df, label)

        clf = svm.LinearSVC()
        clf.fit(X_train,Y_train)
        Y_predict = get_predictions(clf, X_train, Y_train, X_test)


        print(metrics.accuracy_score(Y_true, Y_predict))



if __name__ == '__main__':

    df = read_data(inputfile_path)
    #print(df.shape)
    df_original = df.copy()
    training_df, testing_df = split_training_testing(df)
    SVMClf(training_df, testing_df)