"""
here they be
"""

import numpy as np
import matplotlib.pyplot as plt


# --- Activation Functions and Derivative Calculators --- #
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def sigmoid_back(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))  # It is quicker to calculate this manually rather than call the sigmoid function
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)
    return dZ


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    assert (A.shape == Z.shape)
    return A, cache


def relu_back(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ


# def softmax(Z):
#     shiftZ = Z - np.max(Z)
#     exps = np.exp(shiftZ)
#     A = exps / np.sum(exps)
#     assert (A.shape == Z.shape)
#     return A
#
#
# def softmax_back(dA, cache):
#     Z = cache
#     exps = np.exp(dA - dA.max())
#     dZ = exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
#     assert (dZ.shape == Z.shape)
#     return dZ
