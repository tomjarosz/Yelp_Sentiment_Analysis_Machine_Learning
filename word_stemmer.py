#Author: Tom

from nltk.stem.porter import *
import csv

def word_stemmer(file_name):

    f = open(file_name)
    csv_file = csv.reader(f)

    for review in csv_file[5]:
        for word in review.split(" "):
            PorterStemmer().stem_word(word)
            stem_sentence = " ".join(PorterStemmer().stem_word(word) for word in sentence.split(" "))
        review = stem_sentence
    return csv_file