# Author: Sirui Feng
# This file distills the reviews to public institutions.

import json

public_utilities_path = 'data/public_utilities.json'

with open(public_utilities_path) as datafile:
	b_set =set()
	rv = list()

	for line in datafile:
		row = json.loads(line)
		business_id = row["business_id"]
		if business_id not in b_set:
			rv.append(row)
			b_set.add(business_id)
			if len(b_set)==10:
				break
with open('data/training.txt','w') as f:
	for each in rv:
		print(each, file = f)