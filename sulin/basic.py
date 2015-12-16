from __future__ import division
import pandas as pd
# import seaborn as sb
from sklearn import linear_model, svm, neighbors, metrics, ensemble, preprocessing, tree
import random
import numpy as np

MERCHANDISE = {'jewelry': 0, 'furniture': 1, 'home goods': 2, 'mobile': 3, 'sporting goods': 4, 'office equipment': 5, 'computers': 6,
'books and music': 7, 'auto parts': 8, 'clothing': 9}
COUNTRIES = ['mo', 'jp', 'kr', 'fo', 'je', 'mp', 'tm', 'mn', 'bs', 'dm']

# def get_data(f):

# 	bids = pd.read_csv(f)

# 	num_bids = {}
# 	countries = {}
# 	ips = {}
# 	merchandise = {}
# 	suspect = {}
# 	auctions = {}
# 	times = {}

# 	for row in bids.itertuples():
# 		user = row[2]
# 		auction = row[3]
# 		merch = row[4]
# 		country = row[7]
# 		ip = row[8]
# 		time = row[6]
# 		num_bids[user] = num_bids.get(user, 0) + 1
# 		if countries.get(user) == None:
# 			countries[user] = [country]
# 		else:
# 			countries[user].append(country)
# 		if ips.get(user) == None:
# 			ips[user] = [ip]
# 		else:
# 			ips[user].append(ip)
# 		if merchandise.get(user) == None:
# 			merchandise[user] = [merch]
# 		else:
# 			merchandise[user].append(merch)
# 		if country in COUNTRIES:
# 			suspect[user] = 1
# 		else:
# 			suspect[user] = 0
# 		if auctions.get(user) == None:
# 			auctions[user] = {}
# 		auctions[user][auction] = auctions[user].get(auction, 0) + 1
# 		if times.get(user) == None:
# 			times[user] = {}
# 		if times[user].get(auction) == None:
# 			times[user][auction] = [time]
# 		else:
# 			times[user][auction].append(time)

# 	return num_bids, countries, ips, merchandise, suspect, auctions, times, bids

def get_data(f):

	features = pd.read_csv(f)

	all_f = []
	time_f = []
	score_f = []
	num_f = []
	bid_num_f = []

# 	row[2] = bidder ID
# row[3-6] = time values
# row[7-8] = scores
# row[9] = nums
# row[10] = scores
# row[11] = nums
# row[12] = scores
# row[13-15] = nums
# row[16-17] = bid nums

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

# def handle_data(countries, ips, merchandise, auctions, times):

# 	"""	Creates dictionaries of the sizes of the sets of countries and IPs."""

# 	countries_card = {}
# 	ips_card = {}
# 	merch_card = {}
# 	aucs = {}
# 	avg_times = {}

# 	for user, countries_list in countries.iteritems():
# 		countries_card[user] = len(set(countries_list))
# 	for user, ips_list in ips.iteritems():
# 		ips_card[user] = len(set(ips_list))
# 	for user, merch_list in merchandise.iteritems():
# 		merch_card[user] = MERCHANDISE[set(merch_list).pop()]
# 	for user, auction_dict in auctions.iteritems():
# 		total_bids = 0
# 		total_auctions = 0
# 		for auc, num in auction_dict.iteritems():
# 			total_bids += num
# 			total_auctions += 1
# 		avg = total_bids / total_auctions
# 		aucs[user] = avg
# 	for user, times_dict in times.iteritems():
# 		user_avg = 0
# 		for auc, times_list in times_dict.iteritems():
# 			avg = 0
# 			for i in xrange(len(times_list)-1):
# 				avg += times_list[i+1] - times_list[i]
# 			avg = avg / len(times_list)
# 			user_avg += avg
# 		user_avg = user_avg / len(times_dict)
# 		avg_times[user] = user_avg

# 	return countries_card, ips_card, merch_card, aucs, avg_times

def classify(num, tuples):

	lr = []
	lsvc = []
	# knn = [[] for x in xrange(8)]
	# rfor = []
	# linear = []
	# rbf = []
	# polynomial = []
	dt = []

	size = 1400

	for m in xrange(num):

		random.shuffle(tuples)

		# print len(tuples)

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

		# logreg = linear_model.LogisticRegression(class_weight='balanced')
		# logreg.fit(X, y)
		# accuracy = logreg.score(test_X, test_y)
		# y_predict = logreg.predict(test_X)
		# auc = metrics.roc_auc_score(test_y, y_predict)
		# f1 = metrics.f1_score(test_y, y_predict)
		# score = metrics.average_precision_score(test_y, y_predict)
		# print "Logistic Regression: %s accuracy, %s AUC" % (accuracy, score)
		# lr.append((accuracy, auc))

		# clf = svm.LinearSVC(class_weight='balanced')
		# clf.fit(X, y)
		# # y_predict = clf.predict(test_X)
		# # auc = metrics.roc_auc_score(test_y, y_predict)
		# # # f1 = metrics.f1_score(test_y, y_predict)
		# # accuracy = clf.score(test_X, test_y)
		# # # score = metrics.average_precision_score(test_y, y_predict)
		# # # print "Linear SVM: %s accuracy, %s AUC" % (accuracy, score)
		# # lsvc.append((accuracy, auc))
		# probs = clf.decision_function(test_X)
		# fpr, tpr, _ = metrics.roc_curve(test_y, probs)

		# for i in range(1,9):
		# 	neigh = neighbors.KNeighborsClassifier(n_neighbors=i)
		# 	neigh.fit(X, y)
		# 	y_predict = neigh.predict(test_X)
		# 	auc = metrics.roc_auc_score(test_y, y_predict)
		# 	accuracy = neigh.score(test_X, test_y)
		# 	# f1 = metrics.f1_score(test_y, y_predict)
		# 	score = metrics.average_precision_score(test_y, y_predict)
		# 	# print "%s-NN: %s accuracy, %s AUC" % (i, accuracy, score)
		# 	knn[(i-1)].append((accuracy, score, auc))

		# neigh = neighbors.KNeighborsClassifier(n_neighbors=1)
		# neigh.fit(X, y)
		# probs = neigh.predict_proba(test_X)[:, 1]
		# fpr, tpr, _ = metrics.roc_curve(test_y, probs)

		# clf = svm.SVC(kernel='linear')
		# clf.fit(X, y)
		# y_predict = clf.predict(test_X)
		# auc = metrics.roc_auc_score(test_y, y_predict)
		# accuracy = clf.score(test_X, test_y)
		# linear.append((accuracy, auc))

		# clf = svm.SVC(kernel='rbf')
		# clf.fit(X, y)
		# y_predict = clf.predict(test_X)
		# auc = metrics.roc_auc_score(test_y, y_predict)
		# accuracy = clf.score(test_X, test_y)
		# rbf.append((accuracy, auc))

		# clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=0)
		# clf.fit(X, y)
		# # y_predict = clf.predict(test_X)
		# # auc = metrics.roc_auc_score(test_y, y_predict)
		# # accuracy = clf.score(test_X, test_y)
		# # dt.append((accuracy, auc))

		rfc = ensemble.RandomForestClassifier(n_estimators=10, class_weight='balanced')
		rfc.fit(X, y)
		# y_predict = rfc.predict(test_X)
		# auc = metrics.roc_auc_score(test_y, y_predict)
		# accuracy = rfc.score(test_X, test_y)
		# # f1 = metrics.f1_score(test_y, y_predict)
		# # score = metrics.average_precision_score(test_y, y_predict)
		# # print "Random Forest: %s accuracy, %s AUC" % (accuracy, score)
		# rfor.append((accuracy, auc))
		probs = rfc.predict_proba(test_X)[:, 1]
		fpr, tpr, _ = metrics.roc_curve(test_y, probs)

	return fpr, tpr

# num_bids, countries, ips, merchandise, suspect, auctions, times, bids = get_data('bids.csv')
# countries_card, ips_card, merch_card, aucs, avg_times = handle_data(countries, ips, merchandise, auctions, times)

# train = pd.read_csv('train.csv')
# bidders = []
# outcomes = []
# none = []
# tuples = []

# for row in train.itertuples():
# 	if num_bids.get(row[1]) != None:			
# 		bidders.append(row[1])
# 		outcomes.append(row[4])
# 	# else:
# 	# 	none.append(row[4])

# for i in xrange(len(bidders)):
# 	user = bidders[i]
# 	# tuples.append(([num_bids[user], countries_card[user], ips_card[user], merch_card[user], suspect[user]], outcomes[i]))
# 	tuples.append(([num_bids[user], aucs[user], suspect[user], ips_card[user], countries_card[user], avg_times[user]], outcomes[i]))

time_f, score_f, num_f, bid_num_f, all_f = get_data('features.csv')

fpr, tpr = classify(1, all_f)

print tuple(fpr)
print tuple(tpr)

# total_acc = 0
# total_auc = 0
# for acc, auc in dt:
# 	total_acc += acc
# 	total_auc += auc
# acc = total_acc / 5
# auc = total_auc / 5
# print "Decision Tree: %s, %s" % (acc, auc)

# total_acc = 0
# total_auc = 0
# for acc, auc in rfor:
# 	total_acc += acc
# 	total_auc += auc
# acc = total_acc / 5
# auc = total_auc / 5
# print "Random Forest: %s, %s" % (acc, auc)

# total_acc = 0
# total_auc = 0
# for acc, auc in polynomial:
# 	total_acc += acc
# 	total_auc += auc
# acc = total_acc / 5
# auc = total_auc / 5
# print "Polynomial: %s, %s" % (acc, auc)

# total_acc = 0
# total_auc = 0
# for acc, auc in lr:
# 	total_acc += acc
# 	total_auc += auc
# acc = total_acc / 5
# auc = total_auc / 5
# print "Logistic Regression: %s, %s" % (acc, auc)

# total_acc = 0
# total_auc = 0
# for acc, auc in lsvc:
# 	total_acc += acc
# 	total_auc += auc
# acc = total_acc / 5
# auc = total_auc / 5
# print "Linear SVM: %s, %s" % (acc, auc)

# total_acc = 0
# total_score = 0
# total_auc = 0
# for acc, score, auc in lsvc:
# 	total_acc += acc
# 	total_score += score
# 	total_auc += auc
# acc = total_acc / 5
# score = total_score / 5
# auc = total_auc / 5
# print "Linear SVC: %s, %s, %s" % (acc, score, auc)

# for i in xrange(len(knn)):
# 	total_acc = 0
# 	total_score = 0
# 	total_auc = 0
# 	for acc, score, auc in knn[i]:
# 		total_acc += acc
# 		total_score += score
# 		total_auc += auc
# 	acc = total_acc / 5
# 	score = total_score / 5
# 	auc = total_auc / 5
# 	print "%s-NN: %s, %s, %s" % (i, acc, score, auc)

# total_acc = 0
# total_score = 0
# total_auc = 0
# for acc, score, auc in rfor:
# 	total_acc += acc
# 	total_score += score
# 	total_auc += auc
# acc = total_acc / 5
# score = total_score / 5
# auc = total_auc / 5
# print "Random Forest: %s, %s, %s" % (acc, score, auc)

