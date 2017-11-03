# -*- coding: utf-8 -*-
"""
Created on

@author: fame
"""

import time
import numpy as np
from sklearn.metrics import confusion_matrix
#loading internal files
from load_mnist import *
import hw1_knn  as mlBasics
import matplotlib.pyplot as plt


def reshape_predict(X_train,y_train,X_test,y_test,confusion=False,k=1):
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    # Test on test data
    # 1) Compute distances:
    dists = mlBasics.compute_euclidean_distances(X_train, X_test)
    print "I am here"

    # 2) Run the code below and predict labels:
    y_test_pred = mlBasics.predict_labels(dists, y_train,k=k)
    if not confusion:
        return np.mean(y_test_pred==y_test)*100
    else:
        return np.mean(y_test_pred==y_test)*100, confusion_matrix(y_test, y_test_pred)

def five_fold_cross(X_train, y_train,k):
    num_folds = 5
    set_size = len(X_train)/num_folds
    accuracy = []
    for i in xrange(0,num_folds):
        x_tests = X_train[i*set_size:]
        y_tests = y_train[i*set_size:]
        x_trains = np.concatenate((X_train[:i*set_size] ,X_train[(i+1)*set_size:]),axis=0)
        # x_train = X_train[:i*set_size] + X_train[(i+1)*set_size:]y
        y_trains = np.concatenate((y_train[:i*set_size] , y_train[(i+1)*set_size:]),axis=0)
        # y_train = y_train[:i*set_size] + y_train[(i+1)*set_size:]
        accuracy.append(reshape_predict(x_trains, y_trains, x_tests, y_tests,confusion=False,k=k))
    return np.mean(np.array(accuracy))



user_input = raw_input('enter question ')
X_train, y_train = load_mnist('training')
X_test, y_test = load_mnist('testing')

if user_input == str(1):
    start_time = time.time()
    accuracy = reshape_predict(X_train,y_train,X_test,y_test)
    print '{0:0.02f}'.format(accuracy), "of test examples classified correctly."
    print("--- %s seconds ---" % (time.time() - start_time))

if user_input == str(2):
    #randomly shuffle X_train and Y_train and then take first 100 required sample.
    s = np.arange(X_train.shape[0])
    np.random.shuffle(s)
    X_train = X_train[s]
    y_train - y_train[s]

    sample_set = {}
    new_X_train = []
    new_Y_train = []


    for i in xrange(0,len(y_train)):
        try:
            count = sample_set[y_train[i]]
            if count >= 100:
                continue
            else:
                new_X_train.append(X_train[i])
                new_Y_train.append(y_train[i])
                sample_set[y_train[i]] = sample_set[y_train[i]] + 1
        except:
            # print len(sample_set)
            sample_set[y_train[i]] = 1
            new_X_train.append(X_train[i])
            new_Y_train.append(y_train[i])

    X_train = np.array(new_X_train)
    y_train = np.array(new_Y_train)

    accuracy, confusion_matrix = reshape_predict(X_train,y_train, X_test, y_test,confusion=True)
    print '{0:0.02f}'.format(accuracy), "of test examples classified correctly."
    print confusion_matrix

if user_input == str(3):
    #randomly shuffle X_train and Y_train and then take first 100 required sample.
    print "executing the k-fold task"
    s = np.arange(X_train.shape[0])
    np.random.shuffle(s)
    X_train = X_train[s]
    y_train - y_train[s]

    sample_set = {}
    new_X_train = []
    new_Y_train = []


    for i in xrange(0,len(y_train)):
        try:
            count = sample_set[y_train[i]]
            if count >= 100:
                continue
            else:
                new_X_train.append(X_train[i])
                new_Y_train.append(y_train[i])
                sample_set[y_train[i]] = sample_set[y_train[i]] + 1
        except:
            # print len(sample_set)
            sample_set[y_train[i]] = 1
            new_X_train.append(X_train[i])
            new_Y_train.append(y_train[i])

    X_train = np.array(new_X_train)
    y_train = np.array(new_Y_train)
    start_time = time.time()
    accuracy = []
    for k in xrange(0,15):
        accuracy.append(five_fold_cross(X_train,y_train,k=k+1))
        print "done with ", k+1
    print accuracy
    print np.mean(np.array(accuracy))
    print("--- %s seconds ---" % (time.time() - start_time))
    k = [x for x in xrange(15)]
    #still need to test the function
    plt.plot(accuracy,k)
    plt.show()
    # [58.358333333333334, 35.806666666666665, 29.791666666666664, 27.41833333333334, 25.994999999999997,
    #  23.239999999999998, 22.66, 21.876666666666665, 20.890000000000001, 19.140000000000001, 18.119999999999997,
    #  17.883333333333333, 17.350000000000001, 16.971666666666668, 16.801666666666666]

