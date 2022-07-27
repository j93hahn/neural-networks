import matplotlib.pyplot as plt
import numpy as np
import torch


def extract_data():
    loss = np.load('experiments/weightinit/zeros/loss.npy')
    params = torch.load('experiments/weightinit/zeros/param.pt')
    gradients = torch.load('experiments/weightinit/zeros/grad.pt')
    breakpoint()


def stitch_movie():
    import imageio


def main():
    extract_data()


if __name__ == '__main__':
    main()
