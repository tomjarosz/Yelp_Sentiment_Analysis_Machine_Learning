import pandas as pd
import json
import numpy as np
import csv
from run_models import *

def populate_dict(filename):

    '''
    Generates dictionary for the classified data.
    '''

    with open(filename, 'r') as csvfile:
        classified_reviews = csv.reader(csvfile)
        next(classified_reviews)

        reviews_dict = {}

        for row in classified_reviews:
            if len(row) != 0:

                business_id = row[2]
                review = row[6]
                complaint = row[7]
                compliments = row[8]
                suggestion_for_user = row[10]
                suggestion_for_business = row[11]

                if business_id not in reviews_dict:
                    reviews_dict[business_id] = dict()
                    reviews_dict[business_id]['complaint'] = list()
                    reviews_dict[business_id]['compliments'] = list() 
                    reviews_dict[business_id]['suggestion_for_user'] = list()  
                    reviews_dict[business_id]['suggestion_for_business'] = list()             

                if complaint != 0:
                    reviews_dict[business_id]['complaint'].append(review)

                if compliments != 0:
                    reviews_dict[business_id]['compliments'].append(review)
                
                if suggestion_for_user != 0:
                    reviews_dict[business_id]['suggestion_for_user'].append(review)

                if suggestion_for_business != 0:
                    reviews_dict[business_id]['suggestion_for_business'].append(review)

    return reviews_dict

if __name__ == '__main__':
    populate_dict('result.csv')


