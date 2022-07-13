import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.stats import kde


test = "1"
os.chdir("../../data/test" + test)


# access individual arrays like A['arr_0']
A = np.load("experiment-A.npz")
B = np.load("experiment-B.npz")
C = np.load("experiment-C.npz")
D = np.load("experiment-D.npz")
E = np.load("experiment-E.npz")
F = np.load("experiment-F.npz")
experiments = [A, B, C, D, E, F]


os.chdir("../../plots/test" + test)


def viz_train_losses(exper):
    fig, ax = plt.subplots()
    for experiment in exper:
        ax.plot(experiment['arr_1']/45)
    plt.title("Average Loss Rates of Six Weight Init Methods during MNIST Training")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss over Mini_Batches of 100")
    plt.legend(["Zeros", "Random", "Gaussian", "He", "Xavier", "Xavier Normed"])
    plt.savefig("training_loss.png")
    #plt.show()


def viz_test_losses(exper):
    fig, ax = plt.subplots()
    for experiment in exper:
        ax.plot(experiment['arr_5'])
    plt.title("Comparing Loss Rates of Six Weight Init Methods on MNIST Test Set")
    plt.xlabel("Sample number in MNIST Test Set")
    plt.ylabel("Log Likelihood Loss for each Sample")
    plt.legend(["Zeros", "Random", "Gaussian", "He", "Xavier", "Xavier Normed"])
    plt.savefig("test_loss.png")
    #plt.show()


# ridge plots modeled after https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
def viz_gradients(experiment):
    color = '#990066'; outline = '#000000'
    weights = experiment['arr_2'] # epochs x 10 x 784
    # experiment['arr_3'] = epochs x 10

    gs = grid_spec.GridSpec(weights.shape[0], ncols=1) # epochs by 1
    fig = plt.figure(figsize=(10,6.5))

    ax_objs = []
    spines = ["top", "right", "left", "bottom"]
    x_low = -0.02; x_high = 0.02

    for epoch in reversed(range(weights.shape[0])):
        data = weights[epoch].reshape(-1, 1).squeeze() # line up all data points in one line
        density = kde.gaussian_kde(data)
        x_d = np.linspace(x_low, x_high, 100)
        y_d = density(x_d)

        ax_objs.append(fig.add_subplot(gs[epoch:epoch+1, 0:]))
        ax_objs[-1].plot(x_d, y_d, color=outline, lw=1)
        ax_objs[-1].fill_between(x_d, y_d, alpha=0.2, color=color)

        ax_objs[-1].set_xlim(x_low, x_high)

        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        ax_objs[-1].set_yticks([])
        if epoch == weights.shape[0] - 1:
            ax_objs[-1].set_xlabel("Distribution of Gradients", fontsize=12)
            ax_objs[-1].set_xticks(np.arange(x_low, x_high + 1e-5, (x_high - x_low) / 5))
        else:
            ax_objs[-1].set_xticks([])

        #if epoch == 0 or epoch == weights.shape[0] - 1:
        #    ax_objs[-1].set_ylabel("Epoch" + str(epoch + 1), fontsize=5)

        if epoch == np.floor(weights.shape[0] / 2) + 3:
            ax_objs[-1].set_ylabel("Epoch", fontsize=12)

        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

    gs.update(hspace=-0.9)
    plt.title("Distribution of Gradients for Zeros Initialization", fontweight='bold', fontsize=15)
    # plt.yticks(ticks=[0, 44], labels=["Epoch 1", "Epoch 45"])
    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    #viz_test_losses(experiments)
    #viz_train_losses(experiments)
    viz_gradients(A)
