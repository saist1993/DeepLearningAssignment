# -*- coding: utf-8 -*-
"""
Created on

@author: fame
"""

import numpy as np
import matplotlib.pyplot as plt

def predict(X, W, b):
    """
    implement the function h(x, W, b) here
    X: N-by-D array of training data
    W: D dimensional array of weights
    b: scalar bias

    Should return a N dimensional array
    """
    return sigmoid(np.dot(X, W) + b)


def sigmoid(a):
    """
    implement the sigmoid here
    a: N dimensional numpy array

    Should return a N dimensional array
    """
    return 1 / (1 + np.exp(-a))


def l2loss(X, y, W, b):
    """
    implement the L2 loss function
    X: N-by-D array of training data
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias

    Should return three variables: (i) the l2 loss: scalar, (ii) the gradient with respect to W, (iii) the gradient with respect to b
    """

    gradient_w = 0  # initializing the gradient w.r.t. weight
    gradient_b = 0  # initializing the gradient w.r.t. bias
    V = predict(X, W, b)    # temp variable

    L_w_b = np.sum(np.square(y - V))  # the l2 loss function value

    for i in xrange(0, X.shape[0]):
        gradient_w += -2 * X[i] * (y[i] - V[i]) * (V[i] - np.square(V[i]))
        gradient_b += -2 * (y[i] - V[i]) * (V[i] - np.square(V[i]))

    return L_w_b, gradient_w, gradient_b


def train(X, y, W, b, num_iters=1000, eta=0.001):
    """
    implement the gradient descent here
    X: N-by-D array of training data
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias
    num_iters: (optional) number of steps to take when optimizing
    eta: (optional)  the stepsize for the gradient descent

    Should return the final values of W and b
    """

    L_p = []    # initializing list of loss values for plotting (i.e. Y-axis)
    num_iters_p = np.arange(0,1000,1)   # time (i.e. X-axis)

    for j in xrange(num_iters):
        L, g_w, g_b = l2loss(X, y, W, b)

        W -= (eta * g_w)    # new weights
        b -= (eta * g_b)    # new bias

        L_p.append(L)   # appending the value of L computed in each iteration to complete the list L_p

    plt.plot(num_iters_p, L_p)
    plt.show()

    return W, b
