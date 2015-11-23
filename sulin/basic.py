from __future__ import division
import pandas as pd
# import seaborn as sb
from sklearn import linear_model, svm, neighbors, cluster

MERCHANDISE = {'jewelry': 0, 'furniture': 1, 'home goods': 2, 'mobile': 3, 'sporting goods': 4, 'office equipment': 5, 'computers': 6,
'books and music': 7, 'auto parts': 8, 'clothing': 9}

def get_data(f):

	bids = pd.read_csv(f)

	num_bids = {}
	countries = {}
	ips = {}
	merchandise = {}

	for row in bids.itertuples():
		user = row[2]
		country = row[7]
		ip = row[8]
		merch = row[4]
		num_bids[user] = num_bids.get(user, 0) + 1
		if countries.get(user) == None:
			countries[user] = [country]
		else:
			countries[user].append(country)
		if ips.get(user) == None:
			ips[user] = [ip]
		else:
			ips[user].append(ip)
		if merchandise.get(user) == None:
			merchandise[user] = [merch]
		else:
			merchandise[user].append(merch)

	return num_bids, countries, ips, merchandise, bids

def handle_data(num_bids, countries, ips, merchandise):

	""" Uses seaborn to visualize the three dictionaries.

	Creates dictionaries of the sizes of the sets of countries and IPs."""

	countries_card = {}
	ips_card = {}
	merch_card = {}

	for user, countries_list in countries.iteritems():
		countries_card[user] = len(set(countries_list))
	for user, ips_list in ips.iteritems():
		ips_card[user] = len(set(ips_list))
	for user, merch_list in merchandise.iteritems():
		merch_card[user] = MERCHANDISE[set(merch_list).pop()]

	# train.loc[:, 'num_bids'] = pd.Series(np.zeros(2013), index=train.index)

	return countries_card, ips_card, merch_card


num_bids, countries, ips, merchandise, bids = get_data('bids.csv')
countries_card, ips_card, merch_card = handle_data(num_bids, countries, ips, merchandise)

train = pd.read_csv("train.csv")
bidders = []
outcomes = []

for row in train.itertuples():
	if num_bids.get(row[1]) != None:
		bidders.append(row[1])
		outcomes.append(row[4])

vectors = [[] for x in xrange(len(bidders))]

for i in xrange(len(bidders)):
	user = bidders[i]
	vectors[i] = [num_bids[user], countries_card[user], ips_card[user], merch_card[user]]

print len(bidders)

X = vectors[:1400]
test_X = vectors[1400:]
y = outcomes[:1400]
test_y = outcomes[1400:]

# logreg = linear_model.LogisticRegression()
# logreg.fit(X, y)
# score = logreg.score(test_X, test_y)
# print "Logistic Regression: %s" % score

# clf = svm.LinearSVC()
# clf.fit(X, y)
# score = clf.score(test_X, test_y)
# print "Linear SVM: %s" % score

# clf = svm.SVC(kernel='rbf')
# clf.fit(X, y)
# score = clf.score(test_X, test_y)
# print "SVM, RBF kernel: %s" % score

# clf = svm.SVC(kernel='linear')
# clf.fit(X, y)
# score = clf.score(test_X, test_y)
# print "SVM, Linear kernel: %s" % score

# clf = svm.SVC(kernel='poly')
# clf.fit(X, y)
# score = clf.score(test_X, test_y)
# print "SVM, Polynomial kernel: %s" % score

# for i in range(1,7):
# 	neigh = neighbors.KNeighborsClassifier(n_neighbors=i)
# 	neigh.fit(X, y)
# 	score = neigh.score(test_X, test_y)
# 	print "%s-NN: %s" % (i, score)

for i in range(2,8):
	clf = cluster.KMeans(n_clusters=i)
	clf.fit(X, y)
	score = clf.score(test_X)
	print "%s-means: %s" % (i, score)

""" 179 is greatest # of countries
111918 is greatest # of IPs """
