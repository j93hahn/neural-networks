import matplotlib.pyplot as plt
import numpy as np


test = "1"
experiment = "A"
save_img = "mlp/data/test" + test + "/experiment-" + experiment + ".png"


def viz():
    ...


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


if __name__ == '__main__':
    viz()
