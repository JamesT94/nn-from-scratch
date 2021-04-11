"""
here they also be
"""

import numpy as np
from helper_functions import *

np.random.seed(1)


# --- Initialisation of data --- #
def initialise_parameters(layer_dimensions):  # Successfully tested with data
    parameters = {}
    L = len(layer_dimensions)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dimensions[l], layer_dimensions[l - 1]) / np.sqrt(
            layer_dimensions[l - 1])
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
    cost = - np.sum((Y * np.log(AL)) + ((1 - Y) * np.log(1 - AL))) / m  # Cross-entropy cost function
    return cost


# --- Backward Propagation --- #
def linear_backward(dZ, cache):  # Successfully tested with data
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):  # Successfully tested with data
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_back(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_back(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, final_activation):  # Successfully tested with data
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] \
        = linear_activation_backward(dAL, current_cache, final_activation)

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):  # Successfully tested with data
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters['W' + str(l + 1)] - (learning_rate * grads['dW' + str(l + 1)])
        parameters["b" + str(l + 1)] = parameters['b' + str(l + 1)] - (learning_rate * grads['db' + str(l + 1)])

    return parameters


def predict(X, y, parameters, final_activation):
    m = X.shape[1]
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters, final_activation)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p
