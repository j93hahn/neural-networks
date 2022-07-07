import matplotlib.pyplot as plt
import torch





def load_models():
    m1 = torch.load('model-5.pt') # normalized input MNIST data
    m2 = torch.load('model-6.pt') # non-normalized input MNIST data

    m1p = m1.params()
    m2p = m2.params()

    return m1, m2, m1p, m2p


def viz():
    ...


def main():
    m1, m2, m1p, m2p = load_models()
