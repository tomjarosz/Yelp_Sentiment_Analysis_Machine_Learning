# This file generates summary statistics for review dataset.

import json
from textInsighters_business_summary import *

reviews_path = '../yelp_academic_dataset_review.json'
relevant_states = ['PA','NC','IL','AZ','NV','WI']

def reviews_data(list_of_businessids):
	'''
	Parse throught the reviews dataset of Yelp and output a subset of reviews which are of public utilities.
	The function receives a list of business IDs of public utilites.
	It also creates a dictionary of dictionarys of business info in which the outer key is 'business_id'
	and the inner keys are: 'reviews'(list of review IDs), 'review_count'(count of reviews for the Business)
	'one_star', 'two_star', 'three_star', 'four_star', 'five_star' (the count of reviews by star) 
	
	(['text', 'date', 'review_id', 'stars', 'business_id', 'votes', 'user_id', 'type'])
	'''
	with open(reviews_path) as reviews_file, open('public_utilities.json', 'w') as public_utilities_file:

		business_info = {}
		reviews_count   = 0
		
		for line in reviews_file:
			row = json.loads(line)
			business_id = row['business_id']
			review_id = row['review_id']
			user_id = row['user_id']
			stars = row['stars']
			
			#When we find a business we have already seen
			try:
				business_info[business_id]['public_utility'] = False
			#When we see a business for the first time, add an entry to the dictionary
			except KeyError:	
				business_info[business_id] = {}
				business_info[business_id]['public_utility'] = False
				business_info[business_id]['reviews'] = []
			
			if business_id in list_of_businessids:
				#print(business_id)
				business_info[business_id]['public_utility'] = True
				business_info[business_id]['reviews'].append(review_id)
				business_info[business_id]['review_count'] = business_info[business_id].get('review_count',1) + 1

				if stars == 1:
					business_info[business_id]['one_star'] = business_info[business_id].get('one_star',1) + 1 
				elif stars == 2:
					business_info[business_id]['two_star'] = business_info[business_id].get('two_star',1) + 1
				elif stars == 3:
					business_info[business_id]['three_star'] = business_info[business_id].get('three_star',1) + 1
				elif stars == 4:
					business_info[business_id]['four_star'] = business_info[business_id].get('four_star',1) + 1
				else :
					business_info[business_id]['five_star'] = business_info[business_id].get('five_star',1) + 1

				#json.dump(line, public_utilities_file)
				public_utilities_file.write(line)
				#public_utilities_file.write('\n')	
				reviews_count += 1

		#Emptying the dictionary of the business which we think are not public utilities

		for key in list(business_info):
			if business_info[key]['public_utility'] == False:
				business_info.pop(key,None)


		print('No Reviews', reviews_count)
		print(business_info)

public_utilities = public_services()
reviews_data(public_utilities)
