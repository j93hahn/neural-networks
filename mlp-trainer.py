import numpy as np
import modules as m
from data_loaders import mnist
from pudb import set_trace
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import optim as o


# define up here
model_number = "16"
file = "mlp-arch/model-" + model_number + ".pt"
image_loc = "plots/loss_plot_" + model_number + ".png"
grad1_loc = "plots/weight_grad_plot_" + model_number + ".png"
grad2_loc = "plots/bias_grad_plot_" + model_number + ".png"
text_loc = "plots/loss_plot_" + model_number + ".txt"


def trainer(model, loss, optimizer, scheduler, grad_type="Mini-Batch"):
    train_data = mnist.train_images
    train_labels = mnist.train_labels
    model.train()

    if grad_type == "Mini-Batch":
        epochs = 45
        batch_size = 40
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
                curr_batch_data = _data[lower:upper, :]
                curr_batch_labels = _labels[lower:upper]
                breakpoint()
                prediction = model.forward(curr_batch_data / 255)
                actual = np.zeros((batch_size, 10))
                actual[np.arange(0, batch_size), curr_batch_labels] = 1 # NumPy advanced indexing - produce one-hot encodings

                error = loss.forward(prediction, actual)
                errors[t] += error #np.minimum(1000, error)
                model.backward(prediction, loss.backward(actual))
                optimizer.step()
            scheduler.step()
    elif grad_type == "SGD":
        epochs = 1
        T = 100000
        ii = np.arange(0, T)
        errors = np.zeros(T, dtype=np.float64)
        for e in range(epochs):
            print("-- Beginning Training Epoch " + str(e + 1) + " --")
            for t in tqdm(range(T)):
                optimizer.zero_grad()
                j = np.random.randint(0, train_data.shape[0])
                prediction = model.forward(train_data[j][np.newaxis, :] / 255)
                actual = np.zeros((1, 10))
                actual[0, train_labels[j]] = 1
                error = loss.forward(prediction, actual)
                errors[t] += error
                model.backward(prediction, loss.backward(actual))
                optimizer.step() # SGDM optimizer
    elif grad_type == "Batch":
        epochs = 45
        for e in range(epochs):
            # prepare data for new epoch
            rng = np.random.default_rng()
            permute = rng.permutation(train_data.shape[0])
            _data = train_data[permute]
            _labels = train_labels[permute]

            # begin training epoch
            print("-- Beginning Training Epoch " + str(e + 1) + " --")
            optimizer.zero_grad()
            prediction = model.forward(_data / 255)
            actual = np.zeros((_data.shape[0], 10))
            actual[np.arange(0, _data.shape[0]), _labels] = 1

            error = loss.forward(prediction, actual)
            model.backward(prediction, loss.backward(actual))

            optimizer.step()
            scheduler.step()
        torch.save(model, file)
        return 0, 1
    else:
        raise Exception("Unsupported Gradient Descent Type")

    errors = errors / epochs #average errors loss
    torch.save(model, 'mlp/optimal.pt')
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


def visualizer(x, y, grad=False, layer=0):
    # generate plot of errors over each epoch
    if not grad:
        plt.plot(x, y)
        plt.title("Cross Entropy Loss over 1 Epoch for every 1000 Batches")
        plt.xlabel("Number of Batches (Size = 1)")
        plt.ylabel("Cross Entropy Loss for each Example")
        plt.savefig(image_loc)
        plt.show()
        with open(text_loc, 'w') as f:
            f.close()
    else:
        print("Visualizing Gradient Weight Distribution for Layer " + str(layer))
        fig, axs = plt.subplots(2, 1)
        axs[0].hist(x[layer].reshape((-1, 1)).squeeze(), bins=100, density=True, stacked=True)
        weight_mean = np.mean(x[layer].reshape((-1, 1)).squeeze())
        weight_std = np.std(x[layer].reshape((-1, 1)).squeeze())
        axs[1].hist(y[layer], bins=25, stacked=True, density=True)
        bias_mean = np.mean(y[layer])
        bias_std = np.std(y[layer])
        axs[0].set_title("Gradient Weight Distributions, Normalized Input for Layer " + str(layer))
        axs[0].set_xlabel("Gradient Weight Distribution")
        axs[1].set_title("Gradient Bias Distributions, Normalized Input for Layer " + str(layer))
        axs[1].set_xlabel("Gradient Bias Distribution")
        fig.savefig(grad1_loc)
        print("Gradient Weights Mean & STD: " + str(weight_mean) + ", " + str(weight_std))
        print("Gradient Biases Mean & STD: " + str(bias_mean) + ", " + str(bias_std))
        plt.show()


def main():
    #model = m.Sequential(m.Linear(784, 10)) # Linear layer implementation
    model = m.Sequential(
        m.Linear(784, 100),
        m.ReLU(),
        #m.Dropout(p=0.9),
        m.Linear(100, 64),
        m.ReLU(),
        #m.Dropout(p=0.9),
        m.Linear(64, 16),
        m.ReLU(),
        #m.Dropout(p=0.9),
        m.Linear(16, 10)
    )

    loss = m.SoftMaxLoss()

    optimizer = o.SGDM(model.params())
    scheduler = o.lr_scheduler(optimizer, step_size=15)
    ii, errors = trainer(model, loss, optimizer, scheduler, "Mini-Batch")
    print("Training successfully completed, now beginning testing...")

    trained_model = torch.load('mlp/optimal.pt')
    tester(trained_model)
    print("Maximum loss: ", np.max(errors))

    #_, _, gWeights, gBiases = trained_model.params()

    #print("Visualizing Cross Entropy Loss Distribution")
    #visualizer(x=ii, y=errors, grad=False)
    #breakpoint()
    #visualizer(x=np.array(gWeights, dtype=object), y=np.array(gBiases, dtype=object), grad=True, layer=1)



if __name__ == '__main__':
    main()
