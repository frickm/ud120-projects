#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
from tools.email_preprocess import preprocess
from time import time

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


from sklearn.svm import SVC

#########################################################
### your code goes here ###
clf = SVC(kernel='rbf', C=10000)


## reducing the size of the training set

# features_train = features_train[:int(len(features_train) / 100)]
# labels_train = labels_train[:int(len(labels_train) / 100)]

t0 = time()
clf.fit(features_train, labels_train)
print("training time: {}s".format(round(time() - t0, 3)))

# this time we comute the score directly from the classier (compare to naive bayes)
t1 = time()
accuracy = clf.score(features_train, labels_train)
print("compute accuracy time: {}s".format(round(time() - t1, 3)))

t2 = time()
predictions = clf.predict(features_test)
print("prediction time: {}s".format(round(time() - t2, 3)))

print("accuracy: {}".format(accuracy))

# the following is for the questions in the lectures

# ids = [10, 26, 50]

# predictions = clf.predict(features_test)
# val_ = list(((i, val) for (i, val) in enumerate(predictions) if val == 1))
# print(len(val_))
# score = accuracy_score(predictions, labels_test)
#
# print(score)
#########################################################


