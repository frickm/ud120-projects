#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time

from sklearn import tree
from sklearn.metrics import accuracy_score

from tools.email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = tree.DecisionTreeClassifier(min_samples_split=40)
print(len(features_test[0]))

t0 = time()
clf.fit(features_train, labels_train)
print("training time: {}s".format(round(time() - t0, 3)))

score = accuracy_score(labels_test, clf.predict(features_test))
print(score)

t1 = time()
accuracy = clf.score(features_test, labels_test)
print("compute accuracy time: {}s".format(round(time() - t1, 3)))
print("accuracy: {}".format(accuracy))

# t2 = time()
# predictions = clf.predict(features_test)
# print("prediction time: {}s".format(round(time() - t2, 3)))
#
