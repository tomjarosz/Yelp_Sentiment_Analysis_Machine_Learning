import pandas as pd
import json
import numpy as np
import csv
#from run_models import *

def populate_dict(filename):

    with open(filename, 'r') as csvfile:
        classified_reviews = csv.reader(csvfile)
        next(classified_reviews)

        reviews_dict = {}

        for row in classified_reviews:
            #print(row)
            if len(row) != 0:

                business_id = row[3]
                review = row[7]
                complaint = int(row[9])
                compliments = int(row[10])
                suggestion_for_user = int(row[11])
                suggestion_for_business = int(row[12])

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

#if __name__ == '__main__':
#    populate_dict('result.csv')


