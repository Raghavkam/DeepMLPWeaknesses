import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_train = pd.read_csv("mnist_train2.csv")
x_train = data_train.iloc[:, 1:].values / 255.0
y_train = np.eye(10)[data_train.iloc[:, 0].values]

data_test = pd.read_csv("mnist_test2.csv")
x_test = data_test.iloc[:, 1:].values / 255.0
y_test = np.eye(10)[data_test.iloc[:, 0].values]


def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)


def forward_pass(xi, weights, biases):
    activations = [xi.reshape(1, -1)]
    for W, b in zip(weights[:-1], biases[:-1]):
        z = activations[-1].dot(W) + b
        a = relu(z)
        activations.append(a)
    z = activations[-1].dot(weights[-1]) + biases[-1]
    activations.append(z)
    return activations

def backward_pass(acts, yi, weights, biases, lr):
    a_out = acts[-1]
    delta = a_out - yi
    deltas = [delta]
    for l in range(len(weights) - 1, 0, -1):
        d = deltas[-1].dot(weights[l].T) * relu_deriv(acts[l])
        deltas.append(d)
    deltas.reverse()
    for i in range(len(weights)):
        dw = acts[i].T.dot(deltas[i])
        db = deltas[i].sum(axis=0)
        weights[i] -= lr * dw
        biases[i]  -= lr * db

def accuracy(x, y, weights, biases):
    correct = 0
    for xi, yi in zip(x, y):
        out = xi.reshape(1, -1)
        for W, b in zip(weights[:-1], biases[:-1]):
            out = relu(out.dot(W) + b)
        out = out.dot(weights[-1]) + biases[-1]
        if np.argmax(out) == np.argmax(yi):
            correct += 1
    return correct / len(x)




def train_model(n_layers, hidden_size=10, epochs=20, lr=0.001):

    layer_sizes = [784] + [hidden_size] * n_layers + [10]
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        W = np.random.randn(fan_in, layer_sizes[i+1]) * np.sqrt(2.0 / fan_in)
        b = np.zeros(layer_sizes[i+1])
        weights.append(W)
        biases.append(b)

    for _ in range(epochs):
        perm = np.random.permutation(len(x_train))
        for xi, yi in zip(x_train[perm], y_train[perm]):
            acts = forward_pass(xi, weights, biases)
            backward_pass(acts, yi, weights, biases, lr)

    return (
        accuracy(x_train, y_train, weights, biases),
        accuracy(x_test,  y_test,  weights, biases)
    )


layer_counts = list(range(1, 25))  
train_accs, test_accs = [], []

for n in layer_counts:
    print(n)
    tr, te = train_model(n)
    train_accs.append(tr)
    test_accs.append(te)

#train_accs 

plt.figure(figsize=(8, 5))
plt.plot(layer_counts, train_accs, marker='o', label="Train Accuracy")
plt.plot(layer_counts, test_accs, marker='o', label="Test Accuracy")
plt.xlabel("Number of Hidden Layers")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Number of Hidden ReLU Layers")
plt.xticks(layer_counts)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
