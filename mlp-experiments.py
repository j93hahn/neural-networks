import numpy as np
import modules as m
import optim as o
import copy
import sys

from data_loaders import mnist
from tqdm import tqdm


"""
Test number meanings:
1] One linear layer
2] Two-layer MLP (one linear, one ReLU)
3] Three-layer MLP (two linear, one ReLU)
4] Four-layer MLP (two linear, one ReLU, one dropout)
5] Four-layer MLP (two linear, one ReLU, one batchnorm)
6] Seven-layer MLP (four linear, three ReLU)
7] Seven-layer MLP (three linear, two ReLU, two dropout)
8] Ten-layer MLP (four linear, three ReLU, three batchnorm)
9] Sixteen-layer MLP (six linear, five ReLU, five batchnorm)
"""


test = "1"
experiment = "A"
save_array = "mlp/data/test" + test + "/experiment-" + experiment + ".npz"
save_img = "mlp/plots/test" + test + experiment + ".png"


def process_gradients(optimizer, gradients, epochs):
    """
    Input: a Python list of arrays

    Output: n arrays where n = optimizer.paramCount = len(gradients) / epochs
            and each array has shape (E x *) where E = epochs and * is the shape
            of the gradient parameters
    """
    n = int(len(gradients) / epochs)
    if n != optimizer.paramCount:
        sys.exit("Improper value for n")

    result = []
    for i in range(n):
        x = i # check if x-value is correct
        y = []
        while x < len(gradients):
            y.append(gradients[x])
            x += n
        y = np.asarray(y)
        result.append(y)
    return result


def training(model, loss, optimizer, scheduler=None):
    train_data = mnist.train_images
    train_labels = mnist.train_labels
    model.train()

    epochs = 45
    batch_size = 100
    T = int(train_data.shape[0]/batch_size)
    iterations = np.arange(1, epochs + 1)
    errors = np.zeros(epochs, dtype=np.float64)
    gradients = []

    for e in range(epochs):
        # shuffle the data for every epoch
        rng = np.random.default_rng()
        permute = rng.permutation(train_data.shape[0])
        _data = train_data[permute]
        _labels = train_labels[permute]
        print("-- Beginning Training Epoch " + str(e + 1) + " --")
        for t in tqdm(range(T)):
            optimizer.zero_grad()
            # divide dataset into batches
            lower = 0 + batch_size*t
            upper = batch_size + batch_size*t

            # now perform mini-batch gradient descent
            curr_batch_data = _data[lower:upper, :]
            curr_batch_labels = _labels[lower:upper]
            prediction = model.forward(curr_batch_data / 255)
            actual = np.zeros((batch_size, 10))
            actual[np.arange(0, batch_size), curr_batch_labels] = 1

            errors[e] += loss.forward(prediction, actual)
            model.backward(prediction, loss.backward(actual))
            optimizer.step()

            # retrieve gradients at the end of each epoch
            if t == T - 1:
                _, _g = optimizer.state_dict()
                for i in range(len(_g)):
                    for j in range(len(_g[i])):
                        x = copy.deepcopy(_g[i][j])
                        gradients.append(x)
        if scheduler is not None:
            scheduler.step()

    result = process_gradients(optimizer, gradients, epochs)
    return iterations, errors, result


def inference(model, loss):
    model.eval()
    count = 0

    iterations = int(mnist.test_images.shape[0])
    ii = np.zeros(iterations, dtype=np.float64)
    losses = np.zeros(iterations, dtype=np.float64)

    for i in tqdm(range(iterations)):
        prediction = model.forward(mnist.test_images[i][np.newaxis, :] / 255)
        if np.argmax(prediction) == mnist.test_labels[i]:
            count += 1

        actual = np.zeros((1, 10))
        actual[0, mnist.test_labels[i]] = 1
        losses[i] += loss.forward(prediction, actual)
    
    print("Test success rate: " + str(count / 100) + "%")
    return ii, losses


def main():
    # define model configurations
    model = m.Sequential(m.Linear(784, 10))
    loss = m.SoftMaxLoss()
    optimizer = o.SGDM(model.params())
    #scheduler = o.lr_scheduler(optimizer, step_size=15)

    # training
    #iterations, errors, result = training(model, loss, optimizer)
    print("Training successfully completed, now beginning testing...")

    # inference
    ii, losses = inference(model, loss)

    # save data
    np.savez(save_array, iterations, errors, result[0], result[1], ii, losses)


if __name__ == '__main__':
    main()
