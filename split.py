# Author: Sirui Feng
# Sentence split on periods and conjunctions.


'''
This file splits each review on periods and conjuctions.
'''

import re

def split_period(review):
	p = re.compile(r'[^\s\.][^\.\n]+')
	sentence = p.findall(s)
	return sentence

def split_conjunctions(sentence):
	conjuctions = [';', 'for', 'and', 'nor', 'but', 'or', 'yet', 'so']
	clause = re.split('; | and | nor | but | or | yet | so | although | despite | though | however | on the other hand | in contrast ', sentence)
	clause = [x.strip() for x in clause]
	clause = [x for x in clause if len(x) != 0]
	return clause

def go():
	for reiview in d:
		review = review.lower()
		sentence = split_period(review)
		clause = split_conjunctions(sentence)
