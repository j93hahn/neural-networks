import matplotlib.pyplot as plt
import torch


def viz():
    plt.savefig('plots/new.png')


def main():
    model = torch.load('mlp/optimal.pt') # load model first
    print(model.components())

if __name__ == '__main__':
    main()
