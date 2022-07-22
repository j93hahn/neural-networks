import numpy as np
import modules as m
import optim as o
from data_loaders import mnist
from tqdm import tqdm
import torch


"""
For the MNIST dataset, image sizes go from 28x28 to 14x14 to 7x7 then to 1x1.
- at each successive decrease in image size, we apply a pooling layer with stride 2
- 1x1 convolutional filters are just linear layers

- input_channel = 1, but we can double number of channels for each successive pooling layer
"""


def training(model, loss, optimizer, scheduler):
    train_data = mnist.train_images
    train_labels = mnist.train_labels
    model.train()

    epochs = 15
    batch_size = 100
    T = int(train_data.shape[0]/batch_size)
    ii = np.arange(0, T)
    errors = np.zeros(T, dtype=np.float64)

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
            breakpoint()
            curr_batch_data = _data[lower:upper, :].reshape(batch_size, 1, 28, 28)
            curr_batch_labels = _labels[lower:upper]
            prediction = model.forward(curr_batch_data / 255)
            actual = np.zeros((batch_size, 10))
            actual[np.arange(0, batch_size), curr_batch_labels] = 1 # NumPy advanced indexing - produce one-hot encodings

            error = loss.forward(prediction, actual)
            errors[t] += error #np.minimum(1000, error)
            model.backward(prediction, loss.backward(actual))
            optimizer.step()
        scheduler.step()

def inference(model):
    model.eval()
    count = 0
    iterations = int(mnist.test_images.shape[0])
    for i in tqdm(range(iterations)):
        prediction = model.forward(mnist.test_images[i].reshape(1, 1, 28, 28) / 255)
        if np.argmax(prediction) == mnist.test_labels[i]:
            count += 1
    print("Test success rate: " + str(count / 100) + "%")


def main():
    model = m.Sequential(
        m.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
        m.ReLU(),
        m.Flatten2d(),
        m.Linear(in_features=6272, out_features=10)
    )

    loss = m.SoftMaxLoss()
    optimizer = o.SGDM(model.params())
    scheduler = o.lr_scheduler(optimizer, step_size=15)
    training(model, loss, optimizer, scheduler)
    print("Training completed")

    breakpoint()
    inference(model)

if __name__ == '__main__':
    main()