# Author: Sirui Feng
# Sentence split on periods.


'''
This file splits each review on periods and conjuctions.
'''

import re
import json
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import csv
from word_stemmer import word_stemmer

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

def gen_sentences():
	with open(public_utilities_path) as datafile:
		with open('data/full_data.csv', 'w') as outfile:
			writer = csv.DictWriter(outfile, fieldnames = ['review_id', \
				'business_id', 'user_id', 'stars', 'blob_polarity', 'review', \
				'label'])
			writer.writeheader()
			i=0

			for line in datafile:
				i += 1
				print(i)
				row = json.loads(line)
				review = row['text']
				review = review.lower()

				#split only on periods
				sentences = split_period(review)
				for s in sentences:
					blob = TextBlob(s, analyzer = NaiveBayesAnalyzer())
					polarity = blob.polarity
					#s = word_stemmer(s)

					writer.writerow({'review_id':row['review_id'], \
						'business_id': row['business_id'], \
						'user_id':row['user_id'], 'stars':row['stars'], \
						'blob_polarity': polarity, 'review': s})

gen_sentences()