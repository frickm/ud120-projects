#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
from time import time
from sklearn import naive_bayes

from tools.email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = naive_bayes.GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print("training time: {}s".format(round(time() - t0, 3)))

t1 = time()
accuracy = clf.score(features_train, labels_train)
print("compute accuracy time: {}s".format(round(time() - t1, 3)))

t2 = time()
predictions = clf.predict(features_test)
print("prediction time: {}s".format(round(time() - t2, 3)))

print("accuracy: {}".format(accuracy))

#########################################################
### your code goes here ###


#########################################################


