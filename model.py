import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

data = np.array(data)
m, n = data.shape()
np.random.shuffle(data)

data_test = data[0:1000].T
y_test = data_test[0]
x_test = data_test[1:n]

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]

x_train[:, 0].shape

def init_params():
    w1 = np.random.rand(10, 784)
    b1 = np.random.rand(10, 1)
    w2 = np.random.rand(10, 10)
    b2 = np.random.rand(10, 1)
    return w1, b1, w2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return exp(Z) / np.sum(np.exp(Z))

def forward_prop(w1, b1, w2, b2):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z1)
    return z1, a1, z2, a2

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(z1, a1, z2, a2, w2, y):
    m = y.size
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1/m * dz2.dot(a1.T)
    db2 = 1/m + np.sum(dz2, 2)
    dz1 = w2.T.dot(dz2 + deriv_ReLU(z1))
    dw1 = 1/m * dz1.dot(x.T)
    db1 = 1/m * np.sum(dz1, 2)
    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size

def get_predictions(a2):
    return np.argmax(a2, 0)

def gradient_decent(x, y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if (i% 10 == 0):
            print("Iterations: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(a2), y))
    return w1, b1, w2, b2

w1, b1, w2, b2 = gradient_decent(x_train, y_train, 500, 0.1)