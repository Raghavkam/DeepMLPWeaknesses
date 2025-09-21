import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def run():


    data = pd.read_csv("mnist_train2.csv")
    x1 = data.iloc[:, 1:].values / 255.0
    y1 = data.iloc[:, 0].values
    data2 = pd.read_csv("mnist_test2.csv")
    x2 = data2.iloc[:, 1:].values / 255.0
    y2 = data2.iloc[:, 0].values

    num_classes = 10
    y1 = np.eye(num_classes)[y1]
    y2 = np.eye(num_classes)[y2]

    epochs = 20
    lr     = 0.001

    inplen         = 784
    hidden_size    = 64
    output_size    = num_classes
    layer_sizes    = [inplen, hidden_size, hidden_size, hidden_size, hidden_size, output_size]


    weights = []
    biases  = []
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i+1]
        W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
        b = np.zeros(fan_out)
        weights.append(W)
        biases.append(b)

    errlist, acclist = [], []
    weight_first, weight_last = [], []

    for epoch in range(epochs):

        idx = np.random.permutation(len(x1))
        x1, y1 = x1[idx], y1[idx]

        total_error = 0
        for xi, yi in zip(x1, y1):
       
            activations = [xi.reshape(1, -1)]
            for W, b in zip(weights[:-1], biases[:-1]):
                z = activations[-1].dot(W) + b
                a = relu(z)
                activations.append(a)
   
            z_out = activations[-1].dot(weights[-1]) + biases[-1]
            a_out = sigmoid(z_out)
            activations.append(a_out)

            total_error += 0.5 * np.sum((yi - a_out)**2)

    
            delta = (a_out - yi) * sigmoid_deriv(a_out)
            deltas = [delta]


            for l in range(len(weights)-1, 0, -1):
                d = deltas[-1].dot(weights[l].T) * relu_deriv(activations[l])
                deltas.append(d)
            deltas.reverse()  

        
            for i in range(len(weights)):
                dw = activations[i].T.dot(deltas[i])
                db = np.sum(deltas[i], axis=0)
                weights[i] -= lr * dw
                biases[i]  -= lr * db

   
        weight_first.append(weights[0][0,0])
        weight_last.append(weights[-1][0,0])
        errlist.append(total_error)


        acc = accuracy(x2, y2, weights, biases)
        acclist.append(acc)
        print(f"Epoch {epoch:2d} — err {total_error:.2f}, acc {acc:.4f}")


    epochs_axis = list(range(epochs))
    plt.figure(); plt.plot(epochs_axis, weight_first, label="W₁₁"); plt.plot(epochs_axis, weight_last, label="W₅₁"); plt.legend(); plt.title("First vs Last Weight")
    plt.figure(); plt.plot(epochs_axis, errlist); plt.title("Loss"); plt.figure(); plt.plot(epochs_axis, acclist); plt.title("Accuracy")
    plt.show()


def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):

    return (x > 0).astype(float)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(a):

    return a * (1 - a)

def accuracy(x, y, weights, biases):
    correct = 0
    for xi, yi in zip(x, y):
        out = xi.reshape(1,-1)
        for W, b in zip(weights[:-1], biases[:-1]):
            out = relu(out.dot(W) + b)
        out = sigmoid(out.dot(weights[-1]) + biases[-1])
        if np.argmax(out) == np.argmax(yi):
            correct += 1
    return correct / len(x)

if __name__ == "__main__":
    run()
