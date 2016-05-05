# Author: Sirui Feng
# Sentence split on periods and conjunctions.


'''
This file splits each review on periods and conjuctions.
'''

import re
import json
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import csv

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

def get_training():
	with open(public_utilities_path) as datafile:
		with open('data/training_data.csv', 'w') as outfile:
			writer = csv.DictWriter(outfile, fieldnames = ['review_id', \
				'business_id', 'user_id', 'stars', 'blob_polarity', 'review', \
				'label'])
			writer.writeheader()
			i=0

			for line in datafile:
				i+=1
				print(i)
				row = json.loads(line)
				review = row['text']
				review = review.lower()

				#split only on periods
				sentences = split_period(review)
				for s in sentences:
					blob = TextBlob(s, analyzer = NaiveBayesAnalyzer())
					polarity = blob.polarity

					writer.writerow({'review_id':row['review_id'], \
						'business_id': row['business_id'], \
						'user_id':row['user_id'], 'stars':row['stars'], \
						'blob_polarity': polarity, 'review': s})

get_training()