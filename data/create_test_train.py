from readto_pd_df import *
from sklearn.cross_validation import train_test_split


def create_test_train(filename, data_level):
	'''
	Splits a given data set (filename at a certain data_level) into 3 parts:
	-10 percent hidden
	-15 percent test dat
	-10 percent training data
	'''

	df = read(filename, "json objects", data_level)

	data_id = df[data_level]
	data_left, hidden = train_test_split(data_id, test_size = 0.1, random_state = 0)
	data_left, manual_train = train_test_split(data_left, test_size = 0.11, random_state = 0)

	#data_left.to_csv("data_left.csv", header = True)
	#hidden.to_csv("hidden.csv", header = True)
	#manual_train.to_csv("manual_train.csv", header = True)


	return data_left, hidden, manual_train

