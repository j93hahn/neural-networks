from numpy.lib.stride_tricks import sliding_window_view
from module import Module
import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = (i0.reshape(-1, 1) + i1.reshape(1, -1)).astype(int)
    j = (j0.reshape(-1, 1) + j1.reshape(1, -1)).astype(int)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1).astype(int)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=0, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def im2col(_input, stride):
    ...


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
                 stride=1, padding=0, pad_mode="zeros") -> None:
        super().__init__()
        """
        Assume padding, kernel_size, and stride are all integers

        Input has shape (N, C_in, H_in, W_in) and output has shape (N, C_out, H_out, W_out)
            H_out = np.floor((H_in + 2*padding - kernel_size)/self.stride + 1)
            W_out = np.floor((W_in + 2*padding - kernel_size)/self.stride + 1)

        The ratio of in_channels to groups determines the number of input channels
        that will be grouped per filter (or output channel).
        """
        if in_channels % groups != 0:
            raise Exception("Input channels are not divisible by groups")
        if out_channels % groups != 0:
            raise Exception("Output channels are not divisible by groups")
        self.out_channels = out_channels
        self.feature_count = int(in_channels/groups)
        # self.out_shape = np.floor()

        self.kernel_size = kernel_size # integer type
        self.stride = stride # integer type

        if pad_mode not in ["zeros", "reflect", "symmetric", "mean", "median"]:
            raise Exception("Invalid padding mode specified")
        if not isinstance(padding, int):
            raise Exception("Padding is not of an integral type")
        self.padding = padding
        self.pad_mode = "constant" if pad_mode == "zeros" else pad_mode

        # initialize parameters here - modeled after PyTorch
        _k = groups/(in_channels * (kernel_size ** 2))
        self.weights = np.random.uniform(-np.sqrt(_k), np.sqrt(_k), size=(out_channels, self.feature_count, kernel_size, kernel_size))
        self.biases = np.random.uniform(-np.sqrt(_k), np.sqrt(_k), size=out_channels)
        self.gradWeights = np.zeros_like(self.weights)
        self.gradBiases = np.zeros_like(self.biases)

    def forward(self, _input):
        self.out_spatial_dimension = np.floor((_input.shape[-1]+2*self.padding-self.kernel_size)/self.stride + 1).astype(int)
        if self.padding > 0:
            pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
            _input = np.pad(_input, pad_width, mode=self.pad_mode)

        _windows = sliding_window_view(_input, window_shape=(self.kernel_size, self.kernel_size), axis=(-2, -1))[:, :, ::self.stride, ::self.stride]

        X_col = im2col_indices(_input, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)
        W_rows = self.weights.reshape(self.feature_count, -1)
        _output = np.dot(W_rows, X_col) + self.biases[:, np.newaxis]
        _output = _output.reshape(self.weights.shape[0], self.out_spatial_dimension,
                                  self.out_spatial_dimension, _input.shape[0])
        _output = _output.transpose(3, 0, 1, 2)
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


# flatten input vector on all axes except for the batch dimension
class Flatten2d(Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = True

    def forward(self, _input):
        # input vector shape = (N, C, H, W), output vector shape = (N, C * H * W)
        if self.first: # input shape will always be the same across all batches
            self.shape = _input.shape
            self.first = False
        _output = _input.reshape(self.shape[0], -1)
        return _output

    def backward(self, _input, _gradPrev):
        # Unflatten2d - reshape _gradPrev into _input shape and pass backwards
        _gradCurr = _gradPrev.reshape(self.shape)
        return _gradCurr

    def params(self):
        return None, None

    def name(self):
        return "Flatten2d Layer"


def test_conv2d():
    import torch.nn as nn
    import torch
    standard = nn.Conv2d(1, 6, 2, padding=1, stride=1)
    #import torch
    test = Conv2d(1, 6, 2, padding=1, stride=1)
    _input = np.random.randn(1, 1, 5, 5)
    torch_input = torch.randn(1, 1, 5, 5)
    breakpoint()
    s_o = standard.forward(torch_input)
    t_o = test.forward(_input)
    breakpoint()
    test.backward(_input, 0)


def test_flatten2d():
    test = Flatten2d()
    _input = np.random.randint(1, 12, size=(100, 3, 6, 6))
    breakpoint()
    _output = test.forward(_input)
    breakpoint()
    _gradOutput = test.backward(_input, _output)
    assert _gradOutput.shape == _input.shape


if __name__ == '__main__':
    breakpoint()
    x = np.random.randn(5, 1, 10, 10)
    y = im2col_indices(x, 3, 3, 1, 1)
    breakpoint()
    test_conv2d()
    #test_flatten2d()
