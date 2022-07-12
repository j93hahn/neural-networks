import numpy as np
import modules as m
from data_loaders import mnist
import torch
import matplotlib.pyplot as plt


"""
For the MNIST dataset, image sizes go from 28x28 to 14x14 to 7x7 then to 1x1.
- at each successive decrease in image size, we apply a pooling layer with stride 2
- 1x1 convolutional filters are just linear layers

- input_channel = 1, but we can double number of channels for each successive pooling layer
"""


def main():
    model = m.Sequential(
        m.Conv2d(),
        m.ReLU(),
        m.Conv2d(),
        m.Relu(),
        m.MaxPool(),
        m.Conv2d(),
        m.ReLU(),
        m.MaxPool(),
        m.Linear(),
        m.SoftMax()
    )

    loss = m.CrossEntropyLoss()

if __name__ == '__init__':
    main()