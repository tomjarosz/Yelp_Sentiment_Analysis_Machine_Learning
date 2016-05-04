# SUMMARY STATISTICS AND PRECLEANSING FOR BUSINESS DATASET
# MACHINE LEARNING PROJECT: TEXTINSIGHTERS
# AUTHOR: SIRUI FENG, TURAB HASSAN

import json
import operator

business_data_path = '../yelp_academic_dataset_business.json'
user_data_path = '../yelp_academic_dataset_user.json'
review_data_path = '../yelp_academic_dataset_review.json'
public_categories = 'data/cat_public.csv'
relevant_states = ['PA','NC','IL','AZ','NV','WI']
def business_data():
	'''
	Parse throught the business dataset of Yelp and find the summary statistics.
	Each row is dictionary with the following Keys 

	(['categories', 'open', 'full_address', 
	'type', 'latitude', 'stars', 'review_count', 'longitude', 'name', 'attributes', 'neighborhoods',
	'city', 'business_id', 'hours', 'state']) 
	'''
	
	with open(business_data_path) as data_file:    
		count_of_business = 0
		count_of_business_chicago = 0
		type_of_business = set()
		cities_represented = set()
		cities = set()
		for line in data_file:
			row = json.loads(line)
			
			row_type_of_categories = row['categories']
			for category in row_type_of_categories:
				type_of_business.add(category)
			
			city = row['city']
			cities_represented.add(city)
			if city == '':
				count_of_business_chicago += 1
			count_of_business += 1
		
		list_of_categories = list(type_of_business)
		list_of_categories.sort()
		
		list_of_cities = list(cities_represented)
		list_of_cities.sort()

		return type_of_business


#Modify this into a generic function. 
def business_in_city(category):
	'''
	There are actually not a lot of cities but there are problems of spellings and
	subrubs or different areas of the cities are categorized differently
	Generates a dictionary recording the number of business in each city in the db.

	There are 42 cities that have over 100 business. They are:
	
	('Sun City', 106), ('Litchfield Park', 110), ('Sun Prairie', 114), ('Anthem', 123), 
	('Fitchburg', 123), ('Buckeye', 131), ('Maricopa', 143), ('Apache Junction', 157), 
	('Concord', 161), ('Fountain Hills', 177), ('Casa Grande', 181), ('Pineville', 184), 
	('Kitchener', 188), ('Laval', 196), ('Middleton', 208), ('Cave Creek', 229), 
	('Fort Mill', 238), ('Waterloo', 262), ('Urbana', 262), ('Matthews', 346), 
	('Queen Creek', 348), ('Avondale', 386), ('Goodyear', 459), ('Champaign', 462), 
	('Surprise', 587), ('North Las Vegas', 819), ('Karlsruhe', 898), ('Peoria', 929), 
	('Gilbert', 1716), ('Glendale', 1823), ('Madison', 2104), ('Chandler', 2425), 
	('Tempe', 2773), ('Henderson', 2839), ('Mesa', 3190), ('Edinburgh', 3272), 
	('Pittsburgh', 3337), ('Montréal', 3891), ('Scottsdale', 5139), 
	('Charlotte', 5189), ('Phoenix', 10629), ('Las Vegas', 17423)
	'''

	with open(business_data_path) as data_file:
		cat_dict = dict()
		categories = set()
		for line in data_file:
			row = json.loads(line)
			k = row[category]
			if k not in cat_dict:
				cat_dict[k] = 1
			else:
				cat_dict[k] += 1
			categories.add(k)	
		sorted_category = sorted(cat_dict.items(), key=operator.itemgetter(1))
		print(categories)
		print(cat_dict)
		#for key in cat_dict:
		#	print(key, cat_dict[key])
	#return cat_dict

def user_data():
	'''
	Parse throught the user dataset of Yelp and find the summary statistics. Each row is dictionary 
	with the following keys:
	(['average_stars', 'elite', 'compliments', 'type', 'yelping_since', 'fans', 
	'review_count', 'name', 'user_id', 'friends', 'votes']
	'''
	with open(user_data_path) as data_file:

		user_count = 0
		for line in data_file:
			row = json.loads(line)
			user_count += 1
		
		print( 'user_count',user_count )  

def reviews_data():
	'''
	Parse throught the user dataset of Yelp and find the summary statistics. Each row is dictionary 
	with the following keys:

	(['text', 'date', 'review_id', 'stars', 'business_id', 'votes', 'user_id', 'type'])
	'''
	with open(review_data_path) as data_file:

		reviews = 0
		users   = set()
		business = set()
		for line in data_file:
			row = json.loads(line)
			users.add(row['user_id'])
			business.add(row['business_id'])
			reviews += 1

		print('No Users who reviewed', len(users))
		print('No of Business who got reviewd', len(business))
		print('No of reviews', reviews)	

def public_services():
	'''
	Generates a list of business_ids (1943) that belong to a public service category (36).
	And also list of states in which those business occur
	'''

	categories_dict = dict()
	public_business_id = set()
	count = 0

	with open(public_categories) as data_file:
		for line in data_file:
			cat = line.strip()
			categories_dict[cat] = 0

	with open(business_data_path) as data_file:
		bus_list = list()
		for line in data_file:
			row = json.loads(line)

			category_list = row['categories']
			business_id = row['business_id']
			state = row['state']
			if 'Public Services & Government' in category_list:
				categories_dict['Public Services & Government']+=1
			else:
				for cat in category_list:
					if cat in categories_dict:
						categories_dict[cat]+=1
						break

			
			for cat in category_list:
				if cat in categories_dict and state in relevant_states:
					public_business_id.add(business_id)
					count += 1
					break
	print(count)
	#print(public_business_id)				
	return public_business_id

#business_in_city('state')
public_services()