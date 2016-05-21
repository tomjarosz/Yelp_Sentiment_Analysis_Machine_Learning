#Author: Tom

from nltk.stem.porter import *

def word_stemmer(sentence):

    for word in sentence.split(" "):
        PorterStemmer().stem_word(word)
        stem_sentence = " ".join(PorterStemmer().stem_word(word) for word in sentence.split(" "))
    return stem_sentence