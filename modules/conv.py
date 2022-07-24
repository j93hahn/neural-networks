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
    - https://github.com/SkalskiP/ILearnDeepLearning.py/blob/master/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py
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
        _gradCurr = np.zeros_like(_input)
        #_x = 1
        #for e in range(len(_input.shape)):
        #    _x *= _input.shape[e]
        #_gradCurr = np.arange(_x).reshape(_input.shape)

        # how much to pad _gradCurr by? I believe self.kernel_size - 1 to account for "rotated filter"
        _pad = (self.kernel_size - 1) // 2 # if self.padding > 0 else 0
        _gradCurr = Conv2d.pad(_gradCurr, _pad, "constant")
        _rotKernel = np.rot90(self.weights, 2, axes=(-2, -1))

        # inds0 = (h_in, 1, k, 1); inds1 = (1, h_in, 1, k). h_in accounts for stride
        inds0, inds1 = Conv2d.unroll_img_inds(range(0, _gradCurr.shape[-1] - self.kernel_size + 1, self.stride), self.kernel_size)
        x1 = _gradCurr[:, :, inds0, inds1]

        #indsx, indsy = np.ix_(range(_gradPrev.shape[-1]), range(_gradPrev.shape[-1]))
        indsx, indsy = Conv2d.unroll_img_inds(range(0, _gradPrev.shape[-1], self.stride), 1)
        x2 = _gradPrev[:, :, indsx, indsy]

        breakpoint()
        _gradCurr += np.einsum('n o c d p q, o i k l -> n i c d', x2, _rotKernel)
        # now, all you have to do is multiply _gradPrev[:, :, indsx, indsy] by the weights matrix using np.einsum
        # then add that to _gradCurr[:, :, inds0, inds1] and you should be done :)

        #np.add.at(_gradCurr, )

        if self.padding > 0: # remove padding to match _input shape
            _gradCurr = _gradCurr[:, :, _pad:-_pad, _pad:-_pad]

        assert _gradCurr.shape == _input.shape
        return _gradCurr

        """
        for i in range(self.out_spatial_dim): # go down the height dimension
            v_start = self.stride * i
            v_end = v_start + self.kernel_size

            for j in range(self.out_spatial_dim): # go across the width dimension
                h_start = self.stride * j
                h_end = h_start + self.kernel_size
                # sum along out_channels dimension
                _gradCurr[:, :, v_start:v_end, h_start:h_end] += np.sum(self.weights[np.newaxis, :, :, :, :] * _gradPrev[:, :, np.newaxis, i:i+1, j:j+1], axis=1)

                # sum along batch dimension
                self.gradWeights += np.sum(_input[:, np.newaxis, :, v_start:v_end, h_start:h_end] * _gradPrev[:, :, np.newaxis, i:i+1, j:j+1], axis=0)

        # average parameter gradients across batch dimension
        self.gradWeights /= _input.shape[0] # divide by batch amount

        if self.padding > 0:
            _gradCurr = _gradCurr[:, :, self.padding:-self.padding, self.padding:-self.padding]
        """

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
