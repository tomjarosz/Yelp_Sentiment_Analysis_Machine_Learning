# Author: Sirui Feng
# This file performs cross validation on the testing(labeled) dataset.

def cross_validation(clf, X, y):
	k = 7

	evaludation_dict = dict()

	scores = cross_validation.cross_val_score(clf, X, y_true, cv=k)
	print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	y_predicted = cross_validation.cross_val_predict(clf, X, y, cv=k)

	evaludation_dict["accuracy"] = metrics.accuracy_score(y_true, y_predicted)
	evaludation_dict["precision"] = metrics.precision_score(y_true, y_predicted)
	evaludation_dict["roc_auc"] = metrics.roc_auc_score(y_true, y_predicted)


	return evaludation_dict