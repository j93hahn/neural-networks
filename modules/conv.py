from numpy.lib.stride_tricks import sliding_window_view
from einops import rearrange
from .module import Module
import numpy as np


"""
Materials/documentation used to understand convolutional neural networks
1] Vectorization of Convolutions
    - https://cs231n.github.io/convolutional-networks/#conv
    - https://towardsdatascience.com/how-are-convolutions-actually-performed-under-the-hood-226523ce7fbf
    - https://github.com/numpy/numpy/blob/main/numpy/lib/stride_tricks.py

2] Mapping Input Channels to Output Channels
    - https://iksinc.online/2020/05/10/groups-parameter-of-the-convolution-layer/

3] Backpropagation
    - https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
"""
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1,
                 stride=1, padding=0, pad_mode="zeros", init_method="Uniform") -> None:
        super(Conv2d, self).__init__()

        # initialize channel dimensions
        if in_channels % groups != 0:
            raise Exception("Input channels are not divisible by groups")
        if out_channels % groups != 0:
            raise Exception("Output channels are not divisible by groups")
        self.out_channels = out_channels
        self.groups = groups
        self.feature_count = int(in_channels/groups)
        self.out_features = int(out_channels/groups)
        self.out_spatial_dim = -1

        # initialize parameters
        if init_method == "Zero":
            self.weights = np.zeros((out_channels, self.feature_count, kernel_size, kernel_size))
            self.biases = np.zeros(out_channels)
        elif init_method == "Random":
            self.weights = np.random.randn(out_channels, self.feature_count, kernel_size, kernel_size)
            self.biases = np.random.randn(out_channels)
        elif init_method == "Uniform":
            k = groups/(in_channels * (kernel_size ** 2))
            self.weights = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=(out_channels, self.feature_count, kernel_size, kernel_size))
            self.biases = np.random.uniform(-np.sqrt(k), np.sqrt(k), size=out_channels)
        else:
            raise Exception("Initialization technique not recognized.")

        # initialize gradients
        self.gradWeights = np.zeros_like(self.weights)
        self.gradBiases = np.zeros_like(self.biases)

        # initialize hyperparameters - assume padding, kernel size, and stride are integers
        if pad_mode not in ["zeros", "reflect", "symmetric"]:
            raise Exception("Invalid padding mode specified")
        self.pad_mode = "constant" if pad_mode == "zeros" else pad_mode
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, _input):
        # calculate output spatial dimensions
        if self.out_spatial_dim == -1:
            self.out_spatial_dim = np.floor((_input.shape[-1] + 2*self.padding - self.kernel_size)/self.stride + 1).astype(int)
            assert self.out_spatial_dim > 0

        # pad the input if necessary
        _input = Conv2d.pad(_input, self.padding, self.pad_mode) if self.padding > 0 else _input

        # im2col and strides vectorization techniques
        if self.groups == 1:
            _windows = sliding_window_view(_input, window_shape=(self.kernel_size, self.kernel_size), axis=(-2, -1))[:, :, ::self.stride, ::self.stride]
            _windows = rearrange(_windows, 'n c h w k l -> n (c k l) (h w)') # N, C_in, H_out, W_out, k, k -> N, C_in*k*k, H_out*W_out
            _weights = rearrange(self.weights, 'o f k l -> o (f k l)') # f = self.feature_count
            _output = np.einsum('ijk,lj->ilk', _windows, _weights) + self.biases[np.newaxis, :, np.newaxis] # preserve batch dimension
            return _output.reshape(_input.shape[0], self.out_channels, self.out_spatial_dim, self.out_spatial_dim) # reshape to output dimensions
        """
        else: # grouped convolutions
            _yes = []
            for j in range(self.groups):
                a = sliding_window_view(_input, window_shape=(self.kernel_size, self.kernel_size), axis=(-2, -1))[:, self.feature_count*j:self.feature_count*(j+1), ::self.stride, ::self.stride]
                a = a.reshape(_input.shape[0], self.feature_count*(self.kernel_size**2), self.out_spatial_dim**2)
                b = self.weights.reshape(self.out_channels, self.feature_count*(self.kernel_size**2))[self.out_features*j:self.out_features*(j+1), :]
                c = np.einsum('ijk,lj->ilk', a, b) + self.biases[self.out_features*j:self.out_features*(j+1), :]
                _yes.append(c)
            _yes = np.concatenate(_yes, axis=1).reshape(_input.shape[0], self.out_channels, self.out_spatial_dim, self.out_spatial_dim)
            return _yes
        """
        
    def backward(self, _input, _gradPrev):
        # first, pad input vector and _gradCurr
        da_prev = np.zeros_like(_input, dtype=np.float64)
        da_prev_pad = Conv2d.pad(da_prev, self.padding, mode="constant") if self.padding > 0 else _gradCurr
        a_prev_pad = Conv2d.pad(_input, self.padding, mode="constant") if self.padding > 0 else _input
        dz = _gradPrev
        # now, we apply "backwards" convolutions between _inputPad and _gradPrev
        for i in range(self.out_spatial_dim): # go down the height dimension
            v_start = self.stride * i
            v_end = v_start + self.kernel_size

            for j in range(self.out_spatial_dim): # go across the width dimension
                h_start = self.stride * j
                h_end = h_start + self.kernel_size
                # sum along out_channels dimension
                da_prev_pad[:, :, v_start:v_end, h_start:h_end] += np.sum(self.weights[np.newaxis, :, :, :, :] * dz[:, :, np.newaxis, i:i+1, j:j+1], axis=1)

                # sum along batch dimension
                self.gradWeights += np.sum(a_prev_pad[:, np.newaxis, :, v_start:v_end, h_start:h_end] * dz[:, :, np.newaxis, i:i+1, j:j+1], axis=0)

        # average parameter gradients across batch dimension
        self.gradBiases += np.mean(_gradPrev.sum(axis=(-2, -1)), axis=0)[:, np.newaxis]
        self.gradWeights /= _input.shape[0] # divide by batch amount

        if self.padding > 0:
            _gradCurr = da_prev_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return _gradCurr

    def params(self):
        return [self.weights, self.biases], [self.gradWeights, self.gradBiases]

    @staticmethod
    def pad(_input, padding, mode):
        pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))
        return np.pad(_input, pad_width=pad_width, mode=mode)

    def name(self):
        return "Conv2d Layer"


class Flatten2d(Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = True

    def forward(self, _input): # flatten to batch dimension
        return _input.reshape(_input.shape[0], -1)

    def backward(self, _input, _gradPrev):
        return _gradPrev.reshape(_input.shape)

    def params(self):
        return None, None

    def name(self):
        return "Flatten2d Layer"


def test_backward_conv2d():
    #import torch.nn as nn
    #import torch
    standard = Conv2d(in_channels=1, out_channels=8, kernel_size=3, groups=1,
                      padding=0, stride=1)
    _input = np.arange(10*1*5*5).reshape(10, 1, 5, 5)
    _output = standard.forward(_input)
    breakpoint()
    _gradOutput = standard.backward(_input, _output)
