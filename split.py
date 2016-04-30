# Author: Sirui Feng
# Sentence split on periods and conjunctions.


'''
This file splits each review on periods and conjuctions.
'''

import re
import json
from textblob import TextBlob

public_utilities_path = 'data/public_utilities.json'

def split_period(review):
	p = re.compile(r'[^\s\.][^\.\n]+')
	sentences = p.findall(review)
	return sentences

def split_conjunctions(sentence):
	conjuctions = [';', 'for', 'and', 'nor', 'but', 'or', 'yet', 'so']
	clause = re.split('; | and | nor | but | or | yet | so | although | despite | though | however | on the other hand | in contrast ', sentence)
	clause = [x.strip() for x in clause]
	clause = [x for x in clause if len(x) != 0]
	return clause

def go():
	with open(public_utilities_path) as datafile:
		for line in datafile:
			row = json.loads(line)
			review = row["text"]
			review = review.lower()
			print("the original review in lower case is:")
			print(review)
			sentences = split_period(review)
			for sentence in sentences:
				clause = split_conjunctions(sentence)
				print("the clause is:")
				print(clause)
				for c in clause:
					blob = TextBlob(c)
					for sentence in blob.sentences:
						print(sentence)
						print("score:", sentence.sentiment.polarity)
						print()
			break
go()