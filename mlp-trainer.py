import numpy as np
import modules as m
from data_loaders import mnist
from pudb import set_trace
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib


# define up here
model_number = "4"
file = "mlp-arch/model-" + model_number + ".pt"
image_loc = "plots/loss_plot_" + model_number + ".png"
text_loc = "plots/loss_plot_" + model_number + ".txt"


def trainer(model, loss):
    #set_trace()
    model.train()
    batch_size = 1 # SGD
    epochs = 1
    #iterations = int(mnist.train_images.shape[0] / batch_size)
    T = 100000
    ii = np.arange(0, T, 1000)
    errors = np.zeros(int(T / 1000), dtype=np.int64)
    # actual = np.zeros((10, 1))
    #ii = np.arange(0, 60000, 1000)
    #set_trace()

    #errors = []

    for e in range(epochs):
        print("-- Beginning Training Epoch " + str(e + 1) + " --")
        for t in tqdm(range(T)):
            j = np.random.randint(0, 60000)
            #set_trace()
            prediction = model.forward(mnist.train_images[j][:, np.newaxis] / 255)
            actual = np.zeros((10, 1)) # produce one-hot encoding
            actual[mnist.train_labels[j]] = 1

            error = loss.loss(prediction, actual)
            if t % 1000 == 0:
                #print("iteration " + str(i) + " --------")
                #print(prediction)
                #print(denom_sum)
                #print(actual)
                #print(error)
                #ii.append(t)
                errors[int(t / 1000)] += error
            #set_trace()

            model.backward(loss.backward(prediction, actual))
            #set_trace()
            model.update_params(t) # t used for adam optimization

    errors = errors / epochs #average errors loss
    torch.save(model, file)
    return ii, errors
    #set_trace()
    #print(model.layers)



def tester(model):
    model.eval()

    iterations = int(mnist.test_images.shape[0])
    count = 0
    for i in tqdm(range(iterations)):
        prediction = model.forward(mnist.test_images[i][:, np.newaxis] / 255)

        if np.argmax(prediction) == mnist.test_labels[i]:
            count += 1
    print("Test success rate: " + str(count / 100) + "%")


def visualizer(ii, errors):
    # generate plot of errors over each epoch
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
        m.Linear(784, 10),
        #m.ReLU(),
        #m.Linear(16, 16),
        #m.ReLU(),
        #m.Linear(16, 10),
        m.SoftMax()
    )

    loss = m.CrossEntropyLoss()

    ii, errors = trainer(model, loss)

    print("Training successfully completed, now beginning testing...")
    trained_model = torch.load(file)
    tester(trained_model)
    visualizer(ii, errors)


if __name__ == '__main__':
    main()
