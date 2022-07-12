import matplotlib.pyplot as plt
import torch


def viz():
    plt.savefig('plots/new.png')


def main():
    model = torch.load('optimal.pt') # load model first
    breakpoint()
    print(model.components())

if __name__ == '__main__':
    main()
