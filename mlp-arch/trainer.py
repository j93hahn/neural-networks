import numpy as np
from model import model, CrossEntropyLoss
from loader import mnist
from pudb import set_trace


def trainer(model):
    #set_trace()
    model.train
    batch_size = 1 # SGD
    epochs = 1
    iterations = int(mnist.train_images.shape[0] / batch_size)
    loss = CrossEntropyLoss()
    # actual = np.zeros((10, 1))

    for e in range(epochs):
        for i in range(iterations):
            prediction, denom_sum = model.forward(mnist.train_images[i][:, np.newaxis])
            actual = np.zeros((10, 1)) # produce one-hot encoding
            actual[mnist.train_labels[i]] = 1

            error = loss.loss(prediction, actual)
            if i <= 50: # and i <= 150:
                print("iteration " + str(i) + " --------")
                print(prediction)
                print(denom_sum)
                print(actual)
                print(error)

            model.backward(loss.backward(prediction, actual))
            model.update_params(0.01)


    #set_trace()
    print(model.layers)


def tester(model):
    model.eval

    iterations = int(mnist.test_images.shape[0])
    count = 0
    for i in range(iterations):
        prediction = model.forward(mnist.test_images[i][:, np.newaxis])
        if np.argmax(prediction) == mnist.test_labels[i]:
            count += 1
    print(count/iterations)


def main():
    trainer(model)
    #tester(model)


if __name__ == '__main__':
    main()
