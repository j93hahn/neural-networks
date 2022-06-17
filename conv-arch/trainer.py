from loader import DataLoader
from model import Conv2d, ResNet
import numpy as np
import matplotlib.pyplot as plt

def main():
    # data = DataLoader().mnist()
    data = DataLoader().fashion_mnist()

if __name__ == "__main__":
    main()