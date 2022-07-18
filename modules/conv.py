from numpy.lib.stride_tricks import sliding_window_view
from module import Module
from tqdm import tqdm
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
        self.groups = groups
        self.feature_count = int(in_channels/groups)
        self.out_features = int(out_channels/groups)

        if not isinstance(kernel_size, int) or not isinstance(stride, int):
            raise Exception("Kernel size or stride are not of integer type")
        self.kernel_size = kernel_size
        self.stride = stride

        if pad_mode not in ["zeros", "reflect", "symmetric"]:
            raise Exception("Invalid padding mode specified")
        if not isinstance(padding, int):
            raise Exception("Padding is not of an integral type")
        self.padding = padding
        self.pad_mode = "constant" if pad_mode == "zeros" else pad_mode

        # initialize parameters here - modeled after PyTorch
        _k = groups/(in_channels * (kernel_size ** 2))
        self.weights = np.random.uniform(-np.sqrt(_k), np.sqrt(_k), size=(out_channels, self.feature_count, kernel_size, kernel_size))
        self.biases = np.random.uniform(-np.sqrt(_k), np.sqrt(_k), size=out_channels)[:, np.newaxis]
        self.gradWeights = np.zeros_like(self.weights)
        self.gradBiases = np.zeros_like(self.biases)

    def forward(self, _input):
        # determine output spatial dimension
        self.out_dim = np.floor((_input.shape[-1]+2*self.padding-self.kernel_size)/self.stride + 1).astype(int)

        # apply padding onto the input
        if self.padding > 0:
            pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
            _input = np.pad(_input, pad_width, mode=self.pad_mode)

        """
        Algorithm: for each image in the batch, calculate feature_counts = in_channels/groups and
        out_features = out_channels/groups. These two values determine A) how many input channels
        to convolve with each set of filters, and B) how many filters to include in each output
        channel. Once this is done, then for each group, take the number of input channels specified
        in feature_counts, and convolve them with out_features number of filters. This implementation
        relies upon the im2col and strides via vectorization techniques. Repeat this process for the
        number of groups specified, then stack the results together (you can think of groups as
        separate convolutions on different channels - the spatial dimensions are consistent across all
        grouped convolutions). Finally, after processing each image, stack all images together
        to restore the original batch_size dimension from the previous layer.
        """
        _output = []
        for i in range(_input.shape[0]): # process each image individually
            _curr = []
            for j in range(self.groups):
                _windows = sliding_window_view(_input[i], window_shape=(self.kernel_size, self.kernel_size), axis=(-2, -1))[j:j+self.feature_count, ::self.stride, ::self.stride]
                _windows = _windows.reshape(self.feature_count*(self.kernel_size**2), self.out_dim**2)
                _windows = np.dot(self.weights.reshape(self.out_channels, self.feature_count*(self.kernel_size**2))[j:j+self.out_features, :], _windows) + \
                           self.biases[j:j+self.out_features, :]
                _curr.append(_windows)
            _output.append(np.stack(_curr, axis=0).reshape(self.out_channels, self.out_dim, self.out_dim))
        return np.stack(_output, axis=0) # stack images along batch dimension

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


def test_forward_conv2d():
    import torch.nn as nn
    import torch
    for i in tqdm(range(100)):
        # generate 100 random sample architectures to test forward pass
        batch_size = np.random.randint(1, 100)
        in_channels = np.random.randint(1, 20) * 24
        out_channels = np.random.randint(10, 23) * 24
        groups = np.random.choice((1, 2, 3, 4, 6, 8, 12, 24))
        padding = np.random.randint(0, 5)
        stride = np.random.randint(1, 10)
        kernel_size = np.random.randint(15, 22)

        correct = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size,
                            groups=groups, padding=padding, stride=stride)
        testing = Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size,
                            groups=groups, padding=padding, stride=stride)
        x = np.arange(batch_size*in_channels*kernel_size*kernel_size).reshape(
                      batch_size, in_channels, kernel_size, kernel_size)

        correct_output = correct(torch.tensor(x, dtype=torch.float)).detach().numpy()
        testing_output = testing.forward(x)
        assert correct_output.shape == testing_output.shape

    print("Done testing forward pass :)")


def test_backward_conv2d():
    import torch.nn as nn
    import torch


def test_flatten2d():
    test = Flatten2d()
    _input = np.random.randint(1, 12, size=(100, 3, 6, 6))
    breakpoint()
    _output = test.forward(_input)
    breakpoint()
    _gradOutput = test.backward(_input, _output)
    assert _gradOutput.shape == _input.shape


if __name__ == '__main__':
    test_forward_conv2d()
    #test_backward_conv2d()
    #test_flatten2d()
