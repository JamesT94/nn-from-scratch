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


def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)
    return A


def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))
