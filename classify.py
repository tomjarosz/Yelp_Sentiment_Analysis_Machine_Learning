# Author: Sirui Feng
# This file classifies the full dataset with the best-performed classifier on training set.
# Call classify()


def get_predictions(clf, X_train, y_train_true, X_full):
	clf.fit(X_train,y_train_true)
	y_full_predict = clf.predict(X_full)

	return y_full_predict

def classify(best_clf, X_train, y_train_true, X_full):
	best_clf.fit(X_train, y_train_true)
	y_full = get_predictions(best_clf, X_train, y_train_true, X_full)
	return y_full