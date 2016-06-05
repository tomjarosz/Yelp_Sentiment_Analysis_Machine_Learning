import urllib.parse
from . import make_dict
import json


matching_dict = {}
data = make_dict.populate_dict('textinsighters/result.csv') 

def make_matching_dict ():
    with open('textinsighters/yelp_academic_dataset_business.json') as data_file:    
        for line in data_file:
            row = json.loads(line)
            matching_dict[row['name']] = row['business_id']

def context(business_name):
    '''
    Given the business name the user has entered, find the
    complaints, compliments and suggestions associated with
    it and return it to the views
    '''
    #print(matching_dict)
    buss_id = matching_dict[business_name]
    return data[buss_id]

make_matching_dict()