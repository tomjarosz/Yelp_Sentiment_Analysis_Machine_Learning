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

inputfile_path = 'data/training_scored.csv'

labels = ['complaint', 'compliments', 'suggestion for user', 'suggestion for business']
models_dict = {}

complaint_kwords = list(set(open("data/word_list/complaints.txt").read().splitlines()))
compliments_kwords = list(set(open("data/word_list/compliments.txt").read().splitlines()))
suggestions_busn_kwords = list(set(open("data/word_list/suggestion_busn.txt").read().splitlines()))
suggestions_user_kwords = list(set(open("data/word_list/suggestion_user.txt").read().splitlines()))

models_dict['complaint'] = complaint_kwords
models_dict['compliments'] = complaint_kwords
models_dict['suggestion for user'] = suggestions_user_kwords
models_dict['suggestion for business'] = suggestions_user_kwords
models_dict['neutral'] = list()

def read_data(inputfile_path):
    df = pd.read_csv(inputfile_path, encoding = 'cp1252')

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

def vectorize_X(training_df, testing_df, vocabulary, tfidf=True):
    stopwords = get_stopwords()
    vectorizer = CountVectorizer(stop_words=stopwords, ngram_range=(1,2), analyzer='word', vocabulary = vocabulary)
        
    X_train = vectorizer.fit_transform(training_df['stem_review'])
    X_test = vectorizer.transform(testing_df['stem_review'])

    if tfidf:

        tfidf_transformer = TfidfTransformer()
        X_train = tfidf_transformer.fit_transform(X_train)
        X_test = tfidf_transformer.transform(X_test)

    return X_train, X_test

def get_X_and_Y(training_df, testing_df, label):
    
    X_train = training_df['stem_review']

    X_test = testing_df['stem_review']

    vocabulary = models_dict[label]

    X_train, X_test = vectorize_X(training_df, testing_df, vocabulary)
    print("~~~~~~~~~~~~~~~~~~~~~")
    print(X_train.shape)

    Y_train = training_df[label]
    Y_true = testing_df[label]

    print(np.any(np.isnan(Y_train)))

    print(Y_train.describe())

    return X_train, X_test, Y_train, Y_true

def get_predictions(clf, X_train, Y_train, X_test):
    clf.fit(X_train,Y_train)
    Y_predict = clf.predict(X_test)

    return Y_predict

def SVMClf(training_df, testing_df):

    labels = ['complaint', 'suggestion for user', 'compliments', 'neutral', 'suggestion for business']
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