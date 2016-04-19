import json
import operator


def business_data():
	'''
	Parse throught the business dataset of Yelp and find the summary statistics.
	Each row is dictionary with the following Keys (['categories', 'open', 'full_address', 
	'type', 'latitude', 'stars', 'review_count', 'longitude', 'name', 'attributes', 'neighborhoods',
	'city', 'business_id', 'hours', 'state']) 
	'''
	
	with open('yelp_academic_dataset_business.json') as data_file:    
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
			#print(city)
			cities_represented.add(city)
			if city == '':
				count_of_business_chicago += 1
			count_of_business += 1
		
		list_of_categories = list(type_of_business)
		list_of_categories.sort()
		
		list_of_cities = list(cities_represented)
		list_of_cities.sort()

		print( 'count_of_business',count_of_business )
		print( 'count_of_business_chicago', count_of_business_chicago )
		print( 'cities_represented', len(list_of_cities), list_of_cities )
		print( 'list_of_categoreis',len(list_of_categories),list_of_categories )

def business_in_city():
	'''
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

	with open('yelp_academic_dataset_business.json') as data_file:
		cities = dict()
		for line in data_file:
			row = json.loads(line)

			city = row['city']
			if city not in cities:
				cities[city] = 1
			else:
				cities[city] += 1
		sorted_cities = sorted(cities.items(), key=operator.itemgetter(1))
		print(sorted_cities)
def user_data():
	'''
	Parse throught the user dataset of Yelp and find the summary statistics. Each row is dictionary 
	with the following keys:
	(['average_stars', 'elite', 'compliments', 'type', 'yelping_since', 'fans', 
	'review_count', 'name', 'user_id', 'friends', 'votes']
	'''
	with open('yelp_academic_dataset_user.json') as data_file:

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
	with open('yelp_academic_dataset_review.json') as data_file:

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
business_in_city()
			
