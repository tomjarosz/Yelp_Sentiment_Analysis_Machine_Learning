import json
from textInsighters_business_summary import *

reviews_path = '../yelp_academic_dataset_review.json'

def reviews_data(list_of_businessids):
	'''
	Parse throught the user dataset of Yelp and find the summary statistics. Each row is dictionary 
	with the following keys:
	(['text', 'date', 'review_id', 'stars', 'business_id', 'votes', 'user_id', 'type'])
	'''
	with open(reviews_path) as reviews_file:

		business_info = {}
		reviews_count   = 0
		
		for line in reviews_file:
			row = json.loads(line)
			business_id = row['business_id']
			review_id = row['review_id']
			user_id = row['user_id']
			stars = row['stars']
			
			if business_id in list_of_businessids:
				#print(business_id)
				business_info[business_id] = {}
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

				reviews_count += 1


		print('No Reviews', reviews_count)
		print(business_info)

public_utilities = public_services()
reviews_data(public_utilities)
