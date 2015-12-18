from __future__ import division
import pandas as pd
from sklearn import linear_model, svm, neighbors, metrics, ensemble, preprocessing, tree
import random

def get_data(f):

	""" Construct tuples of list of features and outcomes. """

	features = pd.read_csv(f)

	all_f = []
	time_f = []
	score_f = []
	num_f = []
	bid_num_f = []

	for row in features.itertuples():
		time_1 = float(row[3])
		time_2 = float(row[4])
		time_3 = float(row[5])
		time_4 = float(row[6])
		score_1 = float(row[7])
		score_2 = float(row[8])
		score_3 = float(row[10])
		score_4 = float(row[12])
		num_1 = float(row[9])
		num_2 = float(row[11])
		num_3 = float(row[13])
		num_4 = float(row[14])
		num_5 = float(row[15])
		bid_num_1 = float(row[16])
		bid_num_2 = float(row[17])
		outcome = int(row[18])
		time_f.append(([time_1, time_2, time_3, time_4], outcome))
		score_f.append(([score_1, score_2, score_3, score_4], outcome))
		num_f.append(([num_1, num_2, num_3, num_4, num_5], outcome))
		bid_num_f.append(([bid_num_1, bid_num_2], outcome))
		all_f.append(([time_1, time_2, time_3, time_4, score_1, score_2, score_3, score_4,
			num_1, num_2, num_3, num_4, num_5, bid_num_1, bid_num_2], outcome))

	return time_f, score_f, num_f, bid_num_f, all_f

def classify(num, tuples):

	""" Run classification 'num' number of times. Returns sums of accuracy and AUROC numbers
	across all trials. 

	Commented out code for each classifier is the code to calculate class probabilities or
	confidence scores (depending on the classifier) and, based on those, calculate false positive 
	rates and true positive rates to use in plotting an ROC curve. """

	lr = [0, 0]
	lsvc = [0, 0]
	knn = [[0, 0] for x in xrange(8)]
	dt = [0, 0]
	rfor = [0, 0]

	size = 1400

	for m in xrange(num):

		random.shuffle(tuples)

		features = []
		labels = []
		for i in xrange(len(tuples)):
			features.append(tuples[i][0])
			labels.append(tuples[i][1])
		scaler = preprocessing.StandardScaler().fit(features)
		features = scaler.transform(features)
		X = features[:size]
		y = labels[:size]
		test_X = features[size:]
		test_y = labels[size:]

		logreg = linear_model.LogisticRegression(class_weight='balanced')
		logreg.fit(X, y)
		y_predict = logreg.predict(test_X)
		accuracy = logreg.score(test_X, test_y)
		auc = metrics.roc_auc_score(test_y, y_predict)
		lr[0] += accuracy
		lr[1] += auc
		# probs = logreg.predict_proba(test_X)[:, 1]
		# fpr, tpr, _ = metrics.roc_curve(test_y, probs)

		clf = svm.LinearSVC(class_weight='balanced')
		clf.fit(X, y)
		y_predict = clf.predict(test_X)
		accuracy = clf.score(test_X, test_y)
		auc = metrics.roc_auc_score(test_y, y_predict)
		lsvc[0] += accuracy
		lsvc[1] += auc
		# probs = clf.decision_function(test_X)
		# fpr, tpr, _ = metrics.roc_curve(test_y, probs)

		for i in range(1,9):
			neigh = neighbors.KNeighborsClassifier(n_neighbors=i)
			neigh.fit(X, y)
			y_predict = neigh.predict(test_X)
			accuracy = neigh.score(test_X, test_y)
			auc = metrics.roc_auc_score(test_y, y_predict)
			knn[(i-1)][0] += accuracy
			knn[(i-1)][1] += auc

		""" We only plotted ROC curves for 1-NN, which was usually the best-performing of the 
		nearest neighbors classifiers. """

		# neigh = neighbors.KNeighborsClassifier(n_neighbors=1)
		# neigh.fit(X, y)
		# probs = neigh.predict_proba(test_X)[:, 1]
		# fpr, tpr, _ = metrics.roc_curve(test_y, probs)

		clf = tree.DecisionTreeClassifier(class_weight='balanced')
		clf.fit(X, y)
		y_predict = clf.predict(test_X)
		accuracy = clf.score(test_X, test_y)
		auc = metrics.roc_auc_score(test_y, y_predict)
		dt[0] += accuracy
		dt[1] += auc
		# probs = clf.predict_proba(test_X)[:, 1]
		# fpr, tpr, _ = metrics.roc_curve(test_y, probs)

		rfc = ensemble.RandomForestClassifier(class_weight='balanced')
		rfc.fit(X, y)
		y_predict = rfc.predict(test_X)
		accuracy = rfc.score(test_X, test_y)
		auc = metrics.roc_auc_score(test_y, y_predict)
		rfor[0] += accuracy
		rfor[1] += auc
		# probs = rfc.predict_proba(test_X)[:, 1]
		# fpr, tpr, _ = metrics.roc_curve(test_y, probs)

	return lr, lsvc, knn, dt, rfor
	# return fpr, tpr

time_f, score_f, num_f, bid_num_f, all_f = get_data('features.csv')

""" Change all_f to time_f, score_f, and so forth to use different features.
Change num to change the number of trials. """

num = 5

lr, lsvc, knn, dt, rfor = classify(num, all_f)

""" Uncomment to print out false positive and true positive rates for an ROC curve for 
a single classifier. """

# fpr, tpr = classify(num, all_f)
# print tuple(fpr)
# print tuple(tpr)

""" Calculate average accuracies and AUROC scores across trials for each classifier. """

acc = lr[0] / num
auc = lr[1] / num
print "Logistic Regression: %s, %s" % (acc, auc)

acc = lsvc[0] / num
auc = lsvc[1] / num
print "Linear SVM: %s, %s" % (acc, auc)

for i in xrange(len(knn)):
	acc = knn[i][0] / num
	auc = knn[i][1] / num
	print "%s-NN: %s, %s" % (i+1, acc, auc)

acc = dt[0] / num
auc = dt[1] / num
print "Decision Tree: %s, %s" % (acc, auc)

acc = rfor[0] / num
auc = rfor[1] / num
print "Random Forest: %s, %s" % (acc, auc)
