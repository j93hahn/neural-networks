import numpy as np
import modules as m
import torch.nn as nn
import torch

from tqdm import tqdm


def test_forward_conv2d():
    for i in range(100): # generate 100 random sample architectures to test forward pass
        batch_size = np.random.randint(1, 100)
        in_channels = np.random.randint(10, 20)
        out_channels = np.random.randint(10, 20)
        padding = np.random.randint(0, 5)
        stride = np.random.randint(1, 4)
        kernel_size = np.random.randint(1, 9)
        groups = 1

        k = groups/(in_channels * (kernel_size ** 2))
        _w = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(out_channels, in_channels, kernel_size, kernel_size))
        _b = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=out_channels)

        print("----- Test " + str(i+1) + " -----")
        print("Batch size: " + str(batch_size))
        print("In channels: " + str(in_channels))
        print("Out channels: " + str(out_channels))
        print("Padding: " + str(padding))
        print("Stride: " + str(stride))
        print("Kernel size: " + str(kernel_size))

        pyconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, padding=padding, stride=stride)
        pyconv.weight = nn.Parameter(torch.tensor(_w, dtype=torch.float))
        pyconv.bias = nn.Parameter(torch.tensor(_b, dtype=torch.float))

        testing = m.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, _w = _w, _b = _b, padding=padding, stride=stride)
        x = np.random.randn(batch_size*in_channels*kernel_size*kernel_size).reshape(
                      batch_size, in_channels, kernel_size, kernel_size)

        corr = pyconv(torch.tensor(x, dtype=torch.float)).detach().numpy()
        test = testing.forward(x)
        assert corr.shape == test.shape
        assert np.allclose(corr, test, atol=1e-5)
        print(" ")


def test_backward_conv2d():
    for i in tqdm(range(100)):
        batch_size = np.random.randint(1, 100)
        in_channels = np.random.randint(10, 20)
        out_channels = np.random.randint(10, 20)
        padding = 0#np.random.randint(1, 5)
        stride = np.random.randint(1, 6)
        kernel_size = np.random.randint(1, 9)
        groups=1
        in_dim = np.random.randint(15, 25)

        k = groups/(in_channels * (kernel_size ** 2))
        _w = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(out_channels, in_channels, kernel_size, kernel_size))
        _b = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=out_channels)

        #pyconv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        #                    kernel_size=kernel_size, padding=padding, stride=stride)
        #pyconv.weight = nn.Parameter(torch.tensor(_w, dtype=torch.float))
        #pyconv.bias = nn.Parameter(torch.tensor(_b, dtype=torch.float))

        standard = m.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, _w = _w, _b = _b, padding=padding, stride=stride)
        _input = np.random.randn(batch_size, in_channels, in_dim, in_dim)
        _output = standard.forward(_input)
        _gradOutput = standard.backward(_input, _output)
        assert _gradOutput.shape == _input.shape
        #print("----- Test " + str(i+1) + " Completed -----")


if __name__ == '__main__':
    #test_forward_conv2d()
    test_backward_conv2d()
