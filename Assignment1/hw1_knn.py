# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""
 
import numpy as np
import traceback
import collections

 

def compute_distance(X,Y):
    """

    :param X: Input matrix 1
    :param Y: Input matrix 2
    :return: Euclidean distance between two matrix i.e a float value
    """
    return np.sqrt(((X - Y)**2).sum(-1))
    # return np.linalg.norm(X-Y,axis=-1)

def compute_euclidean_distances( X, Y ) :
    """
    Compute the Euclidean distance between two matricess X and Y
    Input:
    X: N-by-D numpy array
    Y: M-by-D numpy array 
    
    Should return dist: M-by-N numpy array   
    """
    dist = np.zeros(shape=(len(Y), len(X)))
    print dist.shape
    for index_1 in xrange(0,len(Y)):
        for index_2 in xrange(0,len(X)):
            # print index_1,index_2
            dist[index_1][index_2] = compute_distance(Y[index_1],X[index_2])
    return dist

def predict_labels( dists, labels, k=1):
    """
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array - M test examples and N training examples
    labels: is a N dimensional numpy array
    
    Should return  pred_labels: M dimensional numpy array
    """
    print dists.shape
    pred_label = []
    for index in dists:
        #Create a tuple and then sort sort the array and select top k candidates
        s = np.argsort(index)
        temp1= index[s]
        temp = labels[s][:k]
        pred_label.append(collections.Counter(temp).most_common()[0][0])
    pred_label = np.array(pred_label)
    return pred_label

