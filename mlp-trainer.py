import numpy as np
import modules as m
import optimizers as o
from data_loaders import mnist
from pudb import set_trace
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib


def alpha_scheduler(iteration):
    # this will be the n(t) function to determine alpha
    ...


def trainer(model, loss):
    #set_trace()
    model.train()
    batch_size = 1 # SGD
    epochs = 1
    iterations = int(mnist.train_images.shape[0] / batch_size)
    # actual = np.zeros((10, 1))
    ii = np.arange(0, 60000, 1000)

    errors = np.zeros(60)

    for e in range(epochs):
        print("-- EPOCH " + str(e + 1) + " --")
        for i in tqdm(range(iterations)):
            prediction = model.forward(mnist.train_images[i][:, np.newaxis])
            actual = np.zeros((10, 1)) # produce one-hot encoding
            actual[mnist.train_labels[i]] = 1

            error = loss.loss(prediction, actual)
            if i % 1000 == 0:
                #print("iteration " + str(i) + " --------")
                #print(prediction)
                #print(denom_sum)
                #print(actual)
                #print(error)
                x = int(i / 1000)
                errors[x] += error

            model.backward(loss.backward(prediction, actual))
            model.update_params(0.1)

    errors = errors / epochs #average errors loss
    torch.save(model, 'mlp-arch/model-4.pt')
    return ii, errors
    #set_trace()
    #print(model.layers)



def tester(model):
    model.eval()

    iterations = int(mnist.test_images.shape[0])
    count = 0
    for i in range(iterations):
        prediction = model.forward(mnist.test_images[i][:, np.newaxis])

        if np.argmax(prediction) == mnist.test_labels[i]:
            count += 1

        if i % 1000 == 0:
            print("iteration " + str(i) + " --------")
    print(count/iterations)


def visualizer(ii, errors):
    # generate plot of errors over each epoch
    plot_num = 4
    image_loc = "plots/loss_plot_" + str(plot_num) + ".png"
    text_loc = "plots/loss_plot_" + str(plot_num) + ".txt"
    plt.plot(ii, errors)
    plt.title("Cross Entropy Loss over 1 Epoch for every 1000 Batches")
    plt.xlabel("Number of Batches (Size = 1)")
    plt.ylabel("Cross Entropy Loss for each Example")
    plt.savefig(image_loc)
    plt.show()
    with open(text_loc, 'w') as f:
        f.close()


def main():
    model = m.Sequential(
        m.Linear(784, 16),
        m.ReLU(),
        m.Dropout(),
        m.Linear(16, 16),
        m.ReLU(),
        m.Linear(16, 10),
        m.SoftMax()
    )

    loss = m.CrossEntropyLoss()

    #set_trace()
    ii, errors = trainer(model, loss)
    print("Starting testing now")
    # set_trace()
    trained_model = torch.load('mlp-arch/model-4.pt')
    tester(trained_model)
    visualizer(ii, errors)


if __name__ == '__main__':
    main()
