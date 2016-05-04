import json

def readjson(json_filename, top_level_id):
	'''
	created to read json objects and turn them into structured dictionary
	'''
	with open(json_filename) as data_file:
		data_dict = {}

		for line in data_file:
			row = json.loads(line)
			top_id_val = row.pop(top_level_id, None)
			data_dict[top_id_val] = row

	return data_dict