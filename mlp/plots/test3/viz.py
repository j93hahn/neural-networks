import matplotlib.pyplot as plt
import numpy as np
import os


test = "3"
os.chdir("../../data/test" + test)


# access individual arrays like A['arr_0']
A = np.load("experiment-A.npz")
B = np.load("experiment-B.npz")
C = np.load("experiment-C.npz")
D = np.load("experiment-D.npz")
E = np.load("experiment-E.npz")
F = np.load("experiment-F.npz")
G = np.load("experiment-G.npz")
H = np.load("experiment-H.npz")
I = np.load("experiment-I.npz")
J = np.load("experiment-J.npz")
K = np.load("experiment-K.npz")
L = np.load("experiment-L.npz")
M = np.load("experiment-M.npz")


os.chdir("../../plots/test" + test)


def viz_losses(experiments):
    fig, ax = plt.subplots()
    batch_sizes = [5, 10, 25, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000, 15000]
    for i in range(len(experiments)):
        ax.plot(experiments[i]['arr_9'], label=f"size: {batch_sizes[i]}")
    plt.title("Comparing Loss Rates of Thirteen Batch Sizes on MNIST Test Set")
    plt.xlabel("Sample number in MNIST Test Set")
    plt.ylabel("Log Likelihood Loss for each Sample")
    plt.legend()
    plt.savefig("test_loss.png")
    plt.show()


if __name__ == '__main__':
    viz_losses([A, B, C, D, E, F, G, H, I, J, K, L, M])
