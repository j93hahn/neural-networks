import numpy as np
import modules as m
from data_loaders import mnist
from pudb import set_trace
import torch


def trainer(model, loss):
    #set_trace()
    model.train
    batch_size = 1 # SGD
    epochs = 14
    iterations = int(mnist.train_images.shape[0] / batch_size)
    # actual = np.zeros((10, 1))

    for e in range(epochs):
        print("############### epoch " + str(e) + " ###############")
        for i in range(iterations):
            prediction = model.forward(mnist.train_images[i][:, np.newaxis])
            actual = np.zeros((10, 1)) # produce one-hot encoding
            actual[mnist.train_labels[i]] = 1

            error = loss.loss(prediction, actual)
            if i % 1000 == 0:
                print("iteration " + str(i) + " --------")
                #print(prediction)
                #print(denom_sum)
                #print(actual)
                print(error)

            model.backward(loss.backward(prediction, actual))
            model.update_params(0.01)

    torch.save(model, 'check.pt')
    #set_trace()
    #print(model.layers)



def tester(model):
    model.eval

    iterations = int(mnist.test_images.shape[0])
    count = 0
    for i in range(iterations):
        prediction = model.forward(mnist.test_images[i][:, np.newaxis])

        if np.argmax(prediction) == mnist.test_labels[i]:
            count += 1

        if i % 1000 == 0:
            print("iteration " + str(i) + " --------")
    print(count/iterations)


def main():
    model = m.Sequential(
        m.Linear(784, 16),
        m.ReLU(),
        m.Linear(16, 16),
        m.ReLU(),
        m.Linear(16, 10),
        m.SoftMax()
    )

    loss = m.CrossEntropyLoss()

    trainer(model, loss)
    print("Starting testing now")
    # set_trace()
    model = torch.load('check.pt')
    tester(model)


if __name__ == '__main__':
    main()
