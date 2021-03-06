import matplotlib.pyplot as plt
import numpy as np
import os


test = "2"
os.chdir("../../data/test" + test)


# access individual arrays like A['arr_0']
A = np.load("experiment-A.npz")
B = np.load("experiment-B.npz")
C = np.load("experiment-C.npz")
D = np.load("experiment-D.npz")
E = np.load("experiment-E.npz")
F = np.load("experiment-F.npz")
G = np.load("experiment-G.npz")


os.chdir("../../plots/test" + test)


def viz_losses(experiments):
    fig, ax = plt.subplots()
    p = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95]
    #breakpoint()
    for i in range(len(experiments)):
        ax.plot(experiments[i]['arr_7'], label=f"p={p[i]}")
    plt.title("Comparing Loss Rates of Seven Dropoff Rates on MNIST Test Set")
    plt.xlabel("Sample number in MNIST Test Set")
    plt.ylabel("Log Likelihood Loss for each Sample")
    plt.legend()
    plt.savefig("test_loss.png")
    plt.show()


if __name__ == '__main__':
    viz_losses([A, B, C, D, E, F, G])
