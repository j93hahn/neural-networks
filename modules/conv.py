from numpy.lib.stride_tricks import sliding_window_view
from einops import rearrange
from .module import Module
import numpy as np


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
        self.feature_count = int(in_channels/groups)
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
            raise Exception("Initialization technique not recognized")

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
        self._inputWindows = sliding_window_view(_input, window_shape=(self.kernel_size, self.kernel_size), axis=(-2, -1))[:, :, ::self.stride, ::self.stride]
        _windows = rearrange(self._inputWindows, 'n c_in h w kh kw -> n h w (c_in kh kw)')
        _weights = rearrange(self.weights, 'c_out c_in kh kw -> c_out (c_in kh kw)')
        _output = np.einsum('n h w q, c q -> n c h w', _windows, _weights) # q is the collapsed dimension
        _biases = rearrange(self.biases, 'c_out -> 1 c_out 1 1')
        return _output + _biases

    def backward(self, _input, _gradPrev):
        # calculate parameter gradients
        self.gradWeights += np.einsum('n i h w k l, n o h w -> o i k l', self._inputWindows, _gradPrev) / _input.shape[0]
        self.gradBiases += np.mean(_gradPrev.sum(axis=(-2, -1)), axis=0)

        # convolve the adjoint with a rotated kernel to produce _gradCurr
        _pad = (self.kernel_size - 1) // 2
        _gradCurr = Conv2d.pad(np.zeros_like(_input), _pad, "constant")
        _rotKernel = np.rot90(self.weights, 2, axes=(-2, -1))

        # each element in _gradPrev corresponds to a square in _gradCurr with length self.kernel_size convolved with the filter
        inds0, inds1 = Conv2d.unroll_img_inds(range(0, _gradCurr.shape[-1] - self.kernel_size + 1, self.stride), self.kernel_size)
        inds2, inds3 = Conv2d.unroll_img_inds(range(0, _gradPrev.shape[-1], self.stride), 1)
        _gradCurr[:, :, inds0, inds1] += np.einsum('n o c d p q, o i k l -> n i c d p q', _gradPrev[:, :, inds2, inds3], _rotKernel)

        if self.padding > 0: # remove padding to match _input shape
            _gradCurr = _gradCurr[:, :, _pad:-_pad, _pad:-_pad]

        return _gradCurr

    def params(self):
        return [self.weights, self.biases], [self.gradWeights, self.gradBiases]

    @staticmethod
    def pad(_input, padding, mode):
        pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))
        return np.pad(_input, pad_width=pad_width, mode=mode)

    @staticmethod # code taken from Haochen Wang - https://github.com/w-hc
    def unroll_img_inds(base_hinds, filter_h, base_winds=None, filter_w=None):
        # assume spatial dimensions are identical
        filter_w = filter_h if filter_w is None else filter_w
        base_winds = base_hinds if base_winds is None else base_winds

        outer_h, outer_w, inner_h, inner_w = np.ix_(
            base_hinds, base_winds, range(filter_h), range(filter_w)
        )

        return outer_h + inner_h, outer_w + inner_w

    def name(self):
        return "Conv2d Layer"


class Flatten2d(Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = True

    def forward(self, _input): # flatten to batch dimension
        return rearrange(_input, 'n c h w -> n (c h w)')

    def backward(self, _input, _gradPrev):
        return _gradPrev.reshape(_input.shape)

    def params(self):
        return None, None

    def name(self):
        return "Flatten2d Layer"
