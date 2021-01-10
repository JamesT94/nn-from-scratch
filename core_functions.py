"""
here they also be
"""

import numpy as np
from helper_functions import *

np.random.seed(3)


# --- Initialisation of data --- #
def initialise_parameters(layer_dimensions):  # Successfully tested with data
    parameters = {}
    L = len(layer_dimensions)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dimensions[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dimensions[l], layer_dimensions[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dimensions[l], 1))

    return parameters


# --- Forward Propagation ---#
def linear_forward(A, W, b):  # Successfully tested with data
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    assert (Z.shape == (W.shape[0], A.shape[1]))

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):  # Successfully tested with data
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters, final_activation):  # Successfully tested with data
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)

    # This is the final activation function, the desired type is defined in the variable 'final_activation'
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], final_activation)
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


# --- Cost Function --- #
def compute_cost(AL, Y):  # Successfully tested with data
    m = Y.shape[1]
    cost = - np.sum((Y * np.log(AL)) + ((1 - Y) * np.log(1 - AL))) / m
    return cost


# --- Backward Propagation --- #
def linear_backward(dZ, cache):

