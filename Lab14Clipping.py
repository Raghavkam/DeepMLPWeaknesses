import numpy as np
import pandas as pd
import math
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

    epochs = 20
    lr = 1  # keep moderate LR since we’ll clip weights

    inplen = 784
    hiddenLayerlen = 64
    targetlen = 10

    # initialize five layers of weights & biases (unchanged)
    firweights    = np.random.randn(inplen, hiddenLayerlen) * 100
    secweights    = np.random.randn(hiddenLayerlen, hiddenLayerlen) * 100
    thirdweights  = np.random.randn(hiddenLayerlen, hiddenLayerlen) * 100
    fourthweights = np.random.randn(hiddenLayerlen, hiddenLayerlen) * 100
    fifthweights  = np.random.randn(hiddenLayerlen, targetlen) * 100

    firbias    = np.zeros(hiddenLayerlen)
    secbias    = np.zeros(hiddenLayerlen)
    thirdbias  = np.zeros(hiddenLayerlen)
    fourthbias = np.zeros(hiddenLayerlen)
    fifthbias  = np.zeros(targetlen)

    errlist      = []
    acclist      = []
    loss_list    = []

    for epoch in range(epochs):
        indices = np.random.permutation(x1.shape[0])
        x1, y1 = x1[indices], y1[indices]

        Etot = 0
        for i in range(len(x1)):
            x = x1[i].reshape(1, -1)
            y = y1[i].reshape(1, -1)

            # FORWARD PASS through 5 layers with sigmoid
            firlayer, seclayer, thirdlayer, fourthlayer, fiftplayer = ff(
                x,
                [
                    firweights.flatten(),
                    secweights.flatten(),
                    thirdweights.flatten(),
                    fourthweights.flatten(),
                    fifthweights.flatten()
                ],
                [
                    firbias.flatten(),
                    secbias.flatten(),
                    thirdbias.flatten(),
                    fourthbias.flatten(),
                    fifthbias.flatten()
                ]
            )

            # BACKWARD PASS through 5 layers and clip weights
            (firweights, secweights, thirdweights, fourthweights, fifthweights,
             firbias, secbias, thirdbias, fourthbias, fifthbias,
             fiftplayer) = backProp(
                x, y,
                firweights, secweights, thirdweights, fourthweights, fifthweights,
                firbias, secbias, thirdbias, fourthbias, fifthbias,
                lr,
                firlayer, seclayer, thirdlayer, fourthlayer, fiftplayer
            )

            Etot += np.sum(0.5 * (y - fiftplayer) ** 2)

        avg_loss = Etot / len(x1)
        loss_list.append(avg_loss)

        acc = accuracy(
            x2, y2,
            firweights, secweights, thirdweights, fourthweights, fifthweights,
            firbias, secbias, thirdbias, fourthbias, fifthbias
        )

        errlist.append(avg_loss)
        acclist.append(acc)

        # print loss and accuracy at each epoch
        # print loss and accuracy at each epoch
        print("Epoch:", epoch, "   Accuracy:", acc,"   Error Total:", avg_loss)

    epochlist = list(range(epochs))

    # Plot accuracy vs epochs
    plt.figure(figsize=(6, 4))
    plt.plot(epochlist, acclist)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Epochs vs. Accuracy")
    plt.legend()

    # Plot loss changes vs epochs
    plt.figure(figsize=(6, 4))
    plt.plot(epochlist, loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Total Error)")
    plt.title("Epochs vs. Loss")
    plt.legend()

    plt.show()

    # after testing is done, print average weight of each layer
    print("\nAverage weight per layer after training/testing:")
    print(" Layer 1:", np.mean(firweights))
    print(" Layer 2:", np.mean(secweights))
    print(" Layer 3:", np.mean(thirdweights))
    print(" Layer 4:", np.mean(fourthweights))
    print(" Layer 5:", np.mean(fifthweights))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivativefunc(x):
    # x is already sigmoid(z)
    return x * (1 - x)

def dot_product(input, weights, stage):
    return [sum([input[x] * weights[x + s * len(input)] for x in range(len(input))])
            for s in range(stage)]

def ff(xv, weights, biases):
    xv = list(xv)
    for i in range(len(weights)):
        in_size  = len(xv[i])
        out_size = len(biases[i])
        w_2d = weights[i].reshape(in_size, out_size)
        z = np.dot(xv[i], w_2d) + biases[i]
        a = sigmoid(z)
        xv.append(a)
    return xv[1], xv[2], xv[3], xv[4], xv[5]

def backProp(x, y,
             firweights, secweights, thirdweights, fourthweights, fifthweights,
             firbias, secbias, thirdbias, fourthbias, fifthbias,
             lr,
             firlayer, seclayer, thirdlayer, fourthlayer, fiftplayer):

    firlayer    = firlayer.reshape(1, -1)
    seclayer    = seclayer.reshape(1, -1)
    thirdlayer  = thirdlayer.reshape(1, -1)
    fourthlayer = fourthlayer.reshape(1, -1)
    fiftplayer  = fiftplayer.reshape(1, -1)
    x = x.reshape(1, -1)

    # compute gradients through layers
    outputError = (fiftplayer - y) * derivativefunc(fiftplayer)
    err4 = np.dot(outputError, fifthweights.T)   * derivativefunc(fourthlayer)
    err3 = np.dot(err4, fourthweights.T)         * derivativefunc(thirdlayer)
    err2 = np.dot(err3, thirdweights.T)          * derivativefunc(seclayer)
    err1 = np.dot(err2, secweights.T)            * derivativefunc(firlayer)

    # update weights & biases
    fifthweights  -= lr * np.dot(fourthlayer.T, outputError)
    fifthbias     -= lr * np.sum(outputError, axis=0)

    fourthweights -= lr * np.dot(thirdlayer.T, err4)
    fourthbias    -= lr * np.sum(err4, axis=0)

    thirdweights  -= lr * np.dot(seclayer.T, err3)
    thirdbias     -= lr * np.sum(err3, axis=0)

    secweights    -= lr * np.dot(firlayer.T, err2)
    secbias       -= lr * np.sum(err2, axis=0)

    firweights    -= lr * np.dot(x.T, err1)
    firbias       -= lr * np.sum(err1, axis=0)

    # clip all weights to [-1, +1]
    firweights    = np.clip(firweights,    -1, 1)
    secweights    = np.clip(secweights,    -1, 1)
    thirdweights  = np.clip(thirdweights,  -1, 1)
    fourthweights = np.clip(fourthweights, -1, 1)
    fifthweights  = np.clip(fifthweights,  -1, 1)

    return (firweights, secweights, thirdweights, fourthweights, fifthweights,
            firbias, secbias, thirdbias, fourthbias, fifthbias,
            fiftplayer)

def accuracy(x2, y2,
             firweights, secweights, thirdweights, fourthweights, fifthweights,
             firbias, secbias, thirdbias, fourthbias, fifthbias):
    correct = 0
    for i in range(len(x2)):
        x = x2[i].reshape(1, -1)
        y = y2[i].reshape(1, -1)

        out = ff(
            x,
            [
                firweights.flatten(),
                secweights.flatten(),
                thirdweights.flatten(),
                fourthweights.flatten(),
                fifthweights.flatten()
            ],
            [
                firbias.flatten(),
                secbias.flatten(),
                thirdbias.flatten(),
                fourthbias.flatten(),
                fifthbias.flatten()
            ]
        )[4]
        pred_label = np.argmax(out)
        true_label = np.argmax(y)
        if pred_label == true_label:
            correct += 1
    return correct / len(x2)


run()
