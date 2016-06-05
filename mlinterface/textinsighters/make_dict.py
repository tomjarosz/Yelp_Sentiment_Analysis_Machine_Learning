import pandas as pd
import json
import numpy as np
import csv

def populate_dict(filename):

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
                    reviews_dict[business_id]=dict()

                if complaint != 0:
                    if 'complaint' in reviews_dict[business_id]:
                        reviews_dict[business_id]['complaint'].append(review)
                    else:
                        reviews_dict[business_id]['complaint'] = list()

                if compliments != 0:
                    if 'compliments' in reviews_dict[business_id]:
                        reviews_dict[business_id]['compliments'].append(review)
                    else:
                        reviews_dict[business_id]['compliments'] = list() 
                
                if suggestion_for_user != 0:
                    if 'suggestion_for_user' in reviews_dict[business_id]:
                        reviews_dict[business_id]['suggestion_for_user'].append(review)
                    else:
                        reviews_dict[business_id]['suggestion_for_user'] = list()  

                if suggestion_for_business != 0:
                    if 'suggestion_for_business' in reviews_dict[business_id]:
                        reviews_dict[business_id]['suggestion_for_business'].append(review)
                    else:
                        reviews_dict[business_id]['suggestion_for_business'] = list()             

    return reviews_dict

if __name__ == '__main__':
    populate_dict('result.csv')


