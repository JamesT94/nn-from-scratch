"""
Pulling together all the core functions and data preparation
"""

import numpy as np
import pandas as pd
from core_functions import *
import matplotlib.pyplot as plt

import h5py


# --- Data setup --- #
def load_data():
    train_dataset = pd.read_csv('datasets/mnist_train.csv')
    train_set_y_orig = np.array(train_dataset['label'])  # your train set labels
    train_set_x_orig = np.array(train_dataset.drop('label', axis=1))  # your train set features

    test_dataset = pd.read_csv('datasets/mnist_test.csv')
    test_set_y_orig = np.array(test_dataset['label'])  # your train set labels
    test_set_x_orig = np.array(test_dataset.drop('label', axis=1))  # your train set features

    classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape))

# # Reshape the training and test examples
# train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
# test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_orig / 255.
test_x = test_x_orig / 255.

train_x = train_x.T
test_x = test_x.T

train_x = train_x[:, :1000]
train_y = train_y[:, :1000]
test_x = test_x[:, :500]
test_y = test_y[:, :500]

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))

# --- Backward Propagation --- #
layers_dims = [784, 20, 7, 5, 1]


def L_layer_model(X, Y, layers_dims, final_activation, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []

    parameters = initialise_parameters(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters, final_activation)

        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches, final_activation)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 10 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

parameters = L_layer_model(train_x, train_y, layers_dims, 'sigmoid', learning_rate=0.0075,
                           num_iterations=2500, print_cost=True)

pred_train = predict(train_x, train_y, parameters, 'sigmoid')
