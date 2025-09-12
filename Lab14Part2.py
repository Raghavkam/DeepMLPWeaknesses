import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run():
    data = pd.read_csv("mnist_train2.csv")
    x1 = data.iloc[:, 1:].values
    y1 = data.iloc[:, 0].values

    data2 = pd.read_csv("mnist_test2.csv")
    x2 = data2.iloc[:, 1:].values
    y2 = data2.iloc[:, 0].values


    x1 = np.array([img.flatten() for img in x1]) / 255.0
    x2 = np.array([img.flatten() for img in x2]) / 255.0
    num_classes = 10
    y1 = np.eye(num_classes)[y1]
    y2 = np.eye(num_classes)[y2]
    

    epochs = 35
    lr = 0.01

    layer_sizes = [784,10,10,10,10,10]

    weights, biases = [], []
    for i in range(len(layer_sizes)-1):
        fan_in = layer_sizes[i]
        W = np.random.randn(fan_in, layer_sizes[i+1]) * np.sqrt(2.0/fan_in)
        b = np.zeros(layer_sizes[i+1])
        weights.append(W)
        biases.append(b)
    err_history, acc_history, w1_history, w_last_history = [], [], [], []

    for epoch in range(epochs):
        indices = np.random.permutation(x1.shape[0])
        x1, y1 = x1[indices], y1[indices]
        total_error = 0

        for xi, yi in zip(x1, y1):
            acts = forward_pass(xi, weights, biases)
            total_error += backward_pass(acts, yi, weights, biases, lr)


        w1_history.append(weights[0][0,0])
        w_last_history.append(weights[-1][0,0])

        err_history.append(total_error)

        acc = accuracy(x2, y2, weights, biases)

        acc_history.append(acc)

        print("Epoch:", epoch, "   Accuracy:", acc,"   Error Total:", total_error)
        
    epochs_axis = range(epochs)
    plt.figure()
    plt.plot(epochs_axis, w1_history, label="W1")
    plt.plot(epochs_axis, w_last_history, label="W_last")
    plt.legend()
    plt.title("First vs Last Weight")
    plt.figure()
    plt.plot(epochs_axis, err_history)
    plt.title("Loss")
    plt.figure()
    plt.plot(epochs_axis, acc_history)
    plt.title("Accuracy")
    plt.show()

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x>0).astype(float)


def forward_pass(xi, weights, biases):
    activations = [xi.reshape(1,-1)]
    for W, b in zip(weights[:-1], biases[:-1]):
        z = activations[-1].dot(W) + b
        a = relu(z)
        activations.append(a)
    z = activations[-1].dot(weights[-1]) + biases[-1]
    a = z
    activations.append(a)
    return activations

def backward_pass(activations, yi, weights, biases, lr):
    a_out = activations[-1]
    delta = a_out - yi
    deltas = [delta]
    for l in range(len(weights)-1, 0, -1):
        d = deltas[-1].dot(weights[l].T) * relu_deriv(activations[l])
        deltas.append(d)
    deltas.reverse()
    for i in range(len(weights)):
        dw = activations[i].T.dot(deltas[i])
        db = deltas[i].sum(axis=0)
        weights[i] -= lr * dw
        biases[i]  -= lr * db
    return 0.5 * ((yi - a_out)**2).sum()

def accuracy(x, y, weights, biases):
    correct = 0
    for xi, yi in zip(x, y):
        out = xi.reshape(1,-1)
        for W, b in zip(weights[:-1], biases[:-1]):
            out = relu(out.dot(W) + b)
        out = out.dot(weights[-1]) + biases[-1]
        if np.argmax(out) == np.argmax(yi):
            correct += 1
    return correct / len(x)
run()
