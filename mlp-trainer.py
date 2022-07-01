import numpy as np
import modules as m
from data_loaders import mnist
from pudb import set_trace
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib


# define up here
model_number = "5"
file = "mlp-arch/model-" + model_number + ".pt"
image_loc = "plots/loss_plot_" + model_number + ".png"
text_loc = "plots/loss_plot_" + model_number + ".txt"


def trainer(model, loss):
    #set_trace()
    data = mnist.train_images
    labelz = mnist.train_labels
    model.train()
    batch_size = 100 # for stochastic gradient descent, batch_size=1
    epochs = 10
    T = int(data.shape[0]/batch_size)
    ii = np.arange(0, T)
    errors = np.zeros(T, dtype=np.int64)

    for e in range(epochs):
        # shuffle the data for every epoch
        rng = np.random.default_rng()
        permute = rng.permutation(data.shape[0])
        data_ = data[permute]
        labelz_ = labelz[permute]
        print("-- Beginning Training Epoch " + str(e + 1) + " --")

        for t in tqdm(range(T)):
            lower = 0 + batch_size*t
            upper = batch_size + batch_size*t

            # now perform, batch gradient descent
            #set_trace()
            train_batch = data_[lower:upper, :]
            train_labs = labelz_[lower:upper]
            prediction = model.forward(train_batch)
            actual = np.zeros((batch_size, 10)) # produce one-hot encoding
            actual[np.arange(0, batch_size), train_labs] = 1

            error = loss.forward(prediction, actual)

                #print("iteration " + str(i) + " --------")
                #print(prediction)
                #print(denom_sum)
                #print(actual)
                #print(error)
                #ii.append(t)
                #print(error)
            errors[t] += error #np.minimum(5, error)
            #set_trace()

            model.backward(prediction, loss.backward(actual))
            #set_trace()
            #model.update_params(t) # t used for adam optimization

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
        prediction = model.forward(mnist.test_images[i][np.newaxis, :] / 255)

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
        m.Linear(784, 16),
        m.ReLU(),
        m.Linear(16, 10)
    )

    loss = m.SoftMaxLoss()

    #set_trace()
    ii, errors = trainer(model, loss)

    print("Training successfully completed, now beginning testing...")
    trained_model = torch.load(file)
    tester(trained_model)
    visualizer(ii, errors)


if __name__ == '__main__':
    main()
