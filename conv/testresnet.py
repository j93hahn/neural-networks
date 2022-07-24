import modules as m
import optim as o
import numpy as np

from data_loaders import mnist
from tqdm import tqdm


def trainer(model, loss, optimizer):
    train_data = mnist.train_images
    train_labels = mnist.train_labels
    model.train()

    epochs = 30
    batch_size = 100
    T = int(train_data.shape[0]/batch_size)

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
            actual[np.arange(0, batch_size), curr_batch_labels] = 1 # NumPy advanced indexing - produce one-hot encodings

            error = loss.forward(prediction, actual)
            model.backward(prediction, loss.backward(actual))
            optimizer.step()
    #torch.save(model, 'mlp/testres.pt')


def tester(model):
    model.eval()

    iterations = int(mnist.test_images.shape[0])
    count = 0
    for i in tqdm(range(iterations)):
        prediction = model.forward(mnist.test_images[i][np.newaxis, :] / 255)

        if np.argmax(prediction) == mnist.test_labels[i]:
            count += 1
    print("Test success rate: " + str(count / 100) + "%")


def main():
    # Testing a 20-layer MLP
    model = m.Sequential(
        m.Linear(784, 100),
        m.ResidualBlock(
            m.Linear(100, 100),
            m.BatchNorm1d(100),
            m.ReLU(),
        ),
        m.ResidualBlock(
            m.Linear(100, 100),
            m.BatchNorm1d(100),
            m.ReLU(),
        ),
        m.ResidualBlock(
            m.Linear(100, 100),
            m.BatchNorm1d(100),
            m.ReLU(),
        ),
        m.ResidualBlock(
            m.Linear(100, 100),
            m.BatchNorm1d(100),
            m.ReLU(),
        ),
        m.ResidualBlock(
            m.Linear(100, 100),
            m.BatchNorm1d(100),
            m.ReLU(),
        ),
        m.ResidualBlock(
            m.Linear(100, 100),
            m.BatchNorm1d(100),
            m.ReLU(),
        ),
        m.Linear(100, 10),
    )
    loss = m.SoftMaxLoss()
    optimizer = o.SGDM(model.params())
    #scheduler = o.lr_scheduler(optimizer, step_size=15)
    trainer(model, loss, optimizer)
    print("Training successfully completed, now beginning testing...")
    #model = torch.load('mlp/testres.pt')
    tester(model)


if __name__ == '__main__':
    main()
