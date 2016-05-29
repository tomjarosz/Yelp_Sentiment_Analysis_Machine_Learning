def get_predictions(clf, X_train, y_train, X_test):
	clf.fit(X_train,y_train)
	y_predict = clf.predict(X_test)

	return y_predict

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

def classify(best_clf, X_train, y_train, X_full):
	best_clf.fit(X_train, y_train)
	y_full = get_predictions(best_clf, X_train, Y_train, X_full)
	return y_full

def output_full_to_dict(df_full):

	df_full_predicted = df_full

	return