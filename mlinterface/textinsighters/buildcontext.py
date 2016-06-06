import urllib.parse
from . import make_dict
# import make_dict
from sklearn.feature_extraction.text import CountVectorizer
import json
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import copy
#import model.py


matching_dict = {}
data = make_dict.populate_dict('textinsighters/result.csv') 
# data = make_dict.populate_dict('../../result.csv')
# print(data)

def get_stopwords():
    '''
    Provides a list of stop words.
    There are 387 stopwords in total.
    '''
    # with open('../../data/word_list/stoplists.csv', 'r') as f:
    with open('textinsighters/stoplists.csv', 'r') as f:

        stopwords = list()
        for line in f:
            stopwords.append(line.strip())

    stopwords = stopwords + ['you', 'I', 'they', 'we', 'our', 'my']
    return stopwords


def make_matching_dict ():
    with open('textinsighters/yelp_academic_dataset_business.json') as data_file:   
    # with open('../../yelp_academic_dataset_business.json') as data_file:    
 
        for line in data_file:
            row = json.loads(line)
            matching_dict[row['name']] = row['business_id']
            #print(matching_dict)

def context_from_bussname(business_name, case):
    '''
    Given the business name the user has entered, find the
    complaints, compliments and suggestions associated with
    it and return it to the views
    '''
    #print(matching_dict)
    busn_id = matching_dict[business_name]
    all_lines = data[busn_id]
    #print('what we get ',all_lines) 
    results_dict = {}

    stopwords = get_stopwords()

    results_dict['complaints'] = all_lines['complaint']
    results_dict['compliments'] = all_lines['compliments']
    results_dict['user_sugg'] = all_lines['suggestion_for_user']
    results_dict['buss_sugg'] = all_lines['suggestion_for_business']
    results_dict_copy = copy.copy(results_dict)
    if case == 1:
        return results_dict['complaints']
    if case == 2:
        return results_dict['compliments']
    if case == 3:
        return results_dict['user_sugg']
    if case == 4:
        return results_dict['buss_sugg']    
    if case == 0:
        for key in results_dict:
            results = results_dict[key]
            rv = []
            if len(results) > 20:
                stopwords = stopwords + business_name.split()
                cv = CountVectorizer(stop_words=stopwords, ngram_range=(1,3), analyzer='word', max_features = 20)
                cv.fit_transform(results)
                cv.fit(results)
                for key_2 in cv.vocabulary_:
                    tag = pos_tag(word_tokenize(key_2))
                    # print(tag)
                    if tag[0][1] == 'NN': # only tracking nouns
                        rv.append(key_2)
                results_dict_copy[key] = rv
        #print("results:", results_dict)
        #print(results_dict_copy)
        return results_dict_copy['complaints'], results_dict_copy['compliments'], results_dict_copy['user_sugg'], results_dict_copy['buss_sugg']

make_matching_dict()
'''
for key in data:
    test = data[key]
    sentences = test['compliments']
    rv = []
    # print(rv)
    # print("length: {}".format(len(rv)))
    if len(sentences) > 20:
        cv = CountVectorizer(stop_words=stopwords, ngram_range=(1,3), analyzer='word', max_features = 20)
        cv.fit_transform(sentences)
        # counts = cv.transform(sentences)
        # print(counts)
        for key in cv.vocabulary_:
            tag = pos_tag(word_tokenize(key))
            # print(tag)
            if tag[0][1] == 'NN':
                rv.append(key)
        # print(rv)
        # print(sentences, "\n")
# context_from_bussname('Animal Elegance')
'''