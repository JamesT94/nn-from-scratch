import numpy as np
import pandas as pd
from core_functions import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df_train = pd.read_csv(filepath)
    print('Loading data from: ' + filepath)

    return df_train

df_train = load_data('datasets/train.csv')

df_train = df_train[['Age', 'Survived']]
print(df_train.head())

df_train.fillna(0, inplace=True)
df_train.info()

# Explore the dataset
m_train = df_train.shape[0]
num_px = df_train.shape[1]

print("Number of training examples: " + str(m_train))
print("df_train shape: " + str(df_train.shape))

X_train, X_test, y_train, y_test = train_test_split(df_train['Age'], df_train['Survived'],
                                                    test_size=0.2, random_state=1)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

X_train = X_train.reshape(X_train.shape[0], 1)
X_test = X_test.reshape(X_test.shape[0], 1)
y_train = y_train.reshape(X_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

X_train = X_train.transpose()
X_test = X_test.transpose()
y_train = y_train.transpose()
y_test = y_test.transpose()

print("shape of X_train", X_train.shape)
print("Shape of Y_train", y_train.shape)
print("Shape of x_test", X_test.shape)
print('Shape of Y_test', y_test.shape)

layer_dims = [1, 15, 10, 5, 1]

def L_layer_model(X, Y, layers_dims, final_activation, learning_rate=0.0075, num_iterations=2500, print_cost=False):
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

parameters = L_layer_model(X_train, y_train, layer_dims, 'sigmoid', learning_rate=0.0075,
                           num_iterations=2500, print_cost=True)

pred_train = predict(X_train, y_train, parameters, 'sigmoid')