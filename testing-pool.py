import numpy as np
import modules as m
import torch.nn as nn
import torch


def test_forward_pool2d():
    for i in range(100): # generate 100 random sample architectures to test forward pass
        batch_size = np.random.randint(1, 100)
        channels = np.random.randint(10, 20)
        spatial_dim = np.random.randint(5, 20) * 2

        print("----- Test " + str(i+1) + " -----")
        print("Batch size: " + str(batch_size))
        print("In channels: " + str(channels))

        x = np.random.randn(batch_size*channels*spatial_dim*spatial_dim).reshape(
                            batch_size, channels, spatial_dim, spatial_dim)
        pyconv = nn.AvgPool2d(kernel_size=2, stride=2)
        testing = m.Pooling2d(kernel_size=2, stride=2, mode="avg")

        corr = pyconv(torch.tensor(x, dtype=torch.float)).detach().numpy()
        test = testing.forward(x)
        assert corr.shape == test.shape
        assert np.allclose(corr, test, atol=1e-5)
        print(" ")


if __name__ == '__main__':
    test_forward_pool2d()
