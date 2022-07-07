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


def trainer(model, loss, grad_type="Mini-Batch"):
    train_data = mnist.train_images
    train_labels = mnist.train_labels
    model.train()

    if grad_type == "Mini-Batch":
        epochs = 25
        batch_size = 100
        T = int(train_data.shape[0]/batch_size)
        ii = np.arange(0, T)
        errors = np.zeros(T, dtype=np.int64)

        for e in range(epochs):
            # shuffle the data for every epoch
            rng = np.random.default_rng()
            permute = rng.permutation(train_data.shape[0])
            _data = train_data[permute]
            _labels = train_labels[permute]
            print("-- Beginning Training Epoch " + str(e + 1) + " --")
            for t in tqdm(range(T)):
                # divide dataset into batches
                lower = 0 + batch_size*t
                upper = batch_size + batch_size*t

                # now perform batch gradient descent
                curr_batch_data = _data[lower:upper, :]
                curr_batch_labels = _labels[lower:upper]
                prediction = model.forward(curr_batch_data)
                actual = np.zeros((batch_size, 10))
                actual[np.arange(0, batch_size), curr_batch_labels] = 1 # NumPy advanced indexing - produce one-hot encodings

                error = loss.forward(prediction, actual)
                errors[t] += np.minimum(1000, error)
                model.backward(prediction, loss.backward(actual))
    elif grad_type == "SGD":
        epochs = 1
        T = 100000
        ii = np.arange(0, T)
        errors = np.zeros(T, dtype=np.float64)
        for e in range(epochs):
            print("-- Beginning Training Epoch " + str(e + 1) + " --")
            for t in tqdm(range(T)):
                breakpoint()
                j = np.random.randint(0, train_data.shape[0])
                prediction = model.forward(train_data[j][np.newaxis, :] / 255)
                actual = np.zeros((1, 10))
                actual[0, train_labels[j]] = 1
                error = loss.forward(prediction, actual)
                errors[t] += error
                model.backward(prediction, loss.backward(actual))
                # optimizer.step(t + 1)
    elif grad_type == "Batch":
        ...
    else:
        raise Exception("Invalid Gradient Descent Type")

    errors = errors / epochs #average errors loss
    torch.save(model, file)
    return ii, errors


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
    print("Maximum loss: ", np.max(errors))


def main():
    #model = m.Linear(784, 10)
    model = m.Sequential(
        m.Linear(784, 16),
        m.ReLU(),
        m.Linear(16, 10)
    )

    loss = m.SoftMaxLoss()
    ii, errors = trainer(model, loss, "SGD")
    print("Training successfully completed, now beginning testing...")

    trained_model = torch.load(file)
    tester(trained_model)
    visualizer(ii, errors)


if __name__ == '__main__':
    main()
