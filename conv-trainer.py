import numpy as np
import modules as m
from data_loaders import mnist
from pudb import set_trace
import torch
import matplotlib.pyplot as plt


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