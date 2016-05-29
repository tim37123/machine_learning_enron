#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','total_payments', 'from_this_person_to_poi', 'total_stock_value', 'expenses', 'long_term_incentive', 'total_cash']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#I'm going to investigate the data to view how it looks in order:
# for name in data_dict:
# 	print name , "	" ,  data_dict[name]['poi'] , "	" , data_dict[name]['total_payments'] , "	" , data_dict[name]['from_this_person_to_poi'] , "	" , data_dict[name]['total_stock_value'] , "	" , data_dict[name]['long_term_incentive']


# print len(data_dict)


### Task 2: Remove outliers
#Kenneth Lay and the bogus "TOTAL" parameter both are outside acceptable range, deleting TOTAL as an outlier but I'll delete Kenneth Lay below
#ONLY IF he's in the training set.
del data_dict['TOTAL']

#Track all names that don't have payments info
to_remove = []
for key in data_dict:
	if data_dict[key]['total_payments'] == 'NaN':
		to_remove.append(key)

#delete anyone without payment info
for name in to_remove:
	del data_dict[name]

### Task 3: Custom Feature
#This feature slightly boosts the f1 and f2 results of the test
for key in data_dict:
	total_cash = 0
	if data_dict[key]['salary'] != 'NaN':
		total_cash = total_cash + data_dict[key]['salary']
	if data_dict[key]['deferral_payments'] != 'NaN':
		total_cash = total_cash + data_dict[key]['deferral_payments']
	if data_dict[key]['expenses'] != 'NaN':
		total_cash = total_cash + data_dict[key]['expenses']
	if data_dict[key]['loan_advances'] != 'NaN':
		total_cash = total_cash + data_dict[key]['loan_advances']
	if data_dict[key]['director_fees'] != 'NaN':
		total_cash = total_cash + data_dict[key]['director_fees']
	if data_dict[key]['long_term_incentive'] != 'NaN':
		total_cash = total_cash + data_dict[key]['long_term_incentive']
	data_dict[key]['total_cash'] = total_cash

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

# I'm going to try the 3 primary methods of fit: Naive bayes, decision tree, SVM
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB

#I've tried manipulating max_features, max_depth, and splitter among others but have not coerced better results than the defaults
clf_tree = tree.DecisionTreeClassifier(max_features=None, max_depth=None, splitter="best")

#No parameters to manipulate for bayes
clf_bayes = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels)

#as mentioned above on line 25, IF Kenneth Lay is in the test set I'm going to delete him.
tracker = 1
for key in features_train:
	if key == "LAY KENNETH L":
		del features_train[key]
		del labels_train[tracker]
	tracker = tracker + 1

#Run my fit using each method
clf_tree.fit(features_train, labels_train)
pred_tree = clf_tree.predict(features_test)

clf_bayes.fit(features_train, labels_train)
pred_bayes = clf_bayes.predict(features_test)

#Now let's see which performed best
print "F1 Performance of decision tree: ", f1_score(labels_test, pred_tree)
print "F1 Performance of naive bayes: ", f1_score(labels_test, pred_bayes)


#The tree performs better than naive bayes so I'm going to choose that for my clf
print "Recall: ", recall_score(labels_test, pred_tree)
print "Precision: ", precision_score(labels_test, pred_tree)

clf = clf_tree
importance = clf.feature_importances_
print "Rank of feature importances: ", clf.feature_importances_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)