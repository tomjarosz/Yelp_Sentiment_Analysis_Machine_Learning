# Machine Learning Project: Textinsighters
# Creates views for our html pages
# Created by: Vi Nguyen, Sirui Feng, Turab Hassan, Tom Jarosz

import json
import operator
import csv

def business_data():
    '''
    Parse throught the business dataset of Yelp and extract the names and Busss Ids.
    Each row is dictionary with the following Keys (['categories', 'open', 'full_address', 
    'type', 'latitude', 'stars', 'review_count', 'longitude', 'name', 'attributes', 'neighborhoods',
    'city', 'business_id', 'hours', 'state']) 
    '''
    mydict = {}
    mydict_2 = {}
    with open('textinsighters/yelp_academic_dataset_business.json') as data_file, open('textinsighters/result.csv') as id_file:    
        spamreader = csv.reader(id_file, delimiter=',')
        next(spamreader)
        for line in spamreader:
            #print('csv', line[2])
            mydict[line[3]] = 'blank'

        for line in data_file:
            row = json.loads(line)
            mydict_2[row['business_id']] = row['name']

        for key in mydict.keys():
            try:
                mydict[key] =  mydict_2[key]
            except:
        
                pass
        rv = list(mydict.values())
        rv.sort() 
        return rv