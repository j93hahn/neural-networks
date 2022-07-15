from turtle import forward
from numpy.lib.stride_tricks import sliding_window_view, as_strided
from module import Module

import numpy as np
import torch.nn as nn
import torch


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1,
                 stride=1, padding=0, pad_mode="zeros") -> None:
        super().__init__()
        """
        Assume padding, kernel_size, and stride are all integers

        Input has shape (N, C_in, H_in, W_in) and output has shape (N, C_out, H_out, W_out).
            H_out = np.floor((H_in + 2*padding - kernel_size)/self.stride + 1)
            W_out = np.floor((W_in + 2*padding - kernel_size)/self.stride + 1)
        """
        if in_channels % groups != 0:
            raise Exception("Input channels are not divisible by groups")
        if out_channels % groups != 0:
            raise Exception("Output channels are not divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.kernel_size = kernel_size
        self.stride = stride

        if pad_mode not in ["zeros", "reflect", "symmetric", "mean", "median"]:
            raise Exception("Invalid padding mode specified")
        if not isinstance(padding, int):
            raise Exception("Padding is not of an integral type")
        self.padding = padding
        self.pad_mode = "constant" if pad_mode == "zeros" else pad_mode

        # initialize parameters here - modeled after PyTorch
        _k = groups/(in_channels * (kernel_size ** 2))
        _sizeW = (out_channels, int(in_channels/groups), kernel_size, kernel_size)
        self.weights = np.random.uniform(-np.sqrt(_k), np.sqrt(_k), size=_sizeW)
        self.biases = np.random.uniform(-np.sqrt(_k), np.sqrt(_k), size=out_channels)
        self.gradWeights = np.zeros_like(self.weights)
        self.gradBiases = np.zeros_like(self.biases)

    def forward(self, _input):
        if self.padding > 0:
            pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
            _input = np.pad(_input, pad_width, mode=self.pad_mode)

        _output = ...
        return _output

    def backward(self, _input, _gradPrev):

        self.gradWeights += ...
        self.gradBiases += ...

        _gradCurr = ...
        return _gradCurr

    def params(self):
        return [self.weights, self.biases], [self.gradWeights, self.gradBiases]

    def name(self):
        return "Conv2d Layer"


"""
The flattening module will flatten the input vector on all axes except the batch
dimension to prepare the numbers for a linear layer (presumably on an image
classification task).
"""
class Flatten2d(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, _input):
        ...

    def backward(self, _input, _gradPrev):
        ...

    def params(self):
        return None, None

    def name(self):
        return "Flatten2d Layer"

def test_conv2d():
    test = Conv2d(10, 8, 2, 1, padding=1)
    _input = np.random.randn(100, 10, 28, 28)
    breakpoint()
    test.forward(_input)
    breakpoint()
    test.backward(_input, 0)


if __name__ == '__main__':
    test_conv2d()
