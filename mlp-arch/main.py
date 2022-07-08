import matplotlib.pyplot as plt
import torch


def viz():
    ...


def main():
    model = torch.load('model-A.pt') # load model first
    breakpoint()
    print(model.components())

if __name__ == '__main__':
    main()