from numpy.lib.stride_tricks import sliding_window_view
from .module import Module
import numpy as np


"""
For all pooling layers, input should be (N x C x H x W)

Key Asumptions:
1) H == W, i.e. _input.shape[2] == _input.shape[3]
2) kernel_size and stride are both integers but do not need to be same value
3) kernel_size <= stride
4) No padding - the pooling can only happen over the given input image plane
5) For min or max pooling, if multiple elements have the same minimum or maximum value
within a kernel (very unlikely), pass the gradients to all elements with the min
or max value
"""
class Pooling2d(Module):
    def __init__(self, kernel_size, stride=None, mode="max", return_indices=True) -> None:
        super().__init__()
        self.kernel_size = kernel_size # input must be an integer
        self.stride = kernel_size if stride == None else stride

        # hard-code assumptions here
        if type(self.kernel_size) != type(self.stride):
            raise Exception("Stride type does not match kernel filter type")
        if not isinstance(self.stride, int):
            raise Exception("Invalid stride input type")
        if self.stride < self.kernel_size:
            raise Exception("Kernel size is larger than the stride")

        self.return_indices = return_indices # if true, return index of max value, necessary for MaxUnpool2d

        if mode not in ["max", "min", "avg"]:
            raise Exception("Invalid pooling mode specified")
        self.mode = mode

    def forward(self, _input):
        """
        Example Calculation:

        Note: Number of grids N = H / stride
        Each grid should go from 0+stride*n to stride*(n+1) from n is in range(0, N)

        Method 1: Use np.ix_ and np.stack to construct the output arrays
        Method 2: Use sliding_window_view to construct the output arrays

        Tried and tested using np.all() - both methods produce equivalent results
        and have similar runtime complexities
        """
        if _input.shape[2] % self.stride:
            raise Exception("Invalid stride shape for input size shape")
        if _input.shape[2] != _input.shape[3]:
            raise Exception("Input spatial dimension axes do not match")

        self.h = int(_input.shape[2] / self.stride) # determines output spatial dimension

        # Method 1
        # grids = [np.ix_(np.arange(self.stride*i, self.stride*i+self.kernel_size),
        #                 np.arange(self.stride*i, self.stride*i+self.kernel_size)) for i in range(self.h)]
        # _pooled = [_input[:, :, grids[i][0], [grids[j][1] for j in range(self.h)]] for i in range(self.h)]

        # Method 2
        _windows = sliding_window_view(_input, window_shape=(self.kernel_size, self.kernel_size), axis=(-2, -1))[:, :, ::self.stride, ::self.stride]

        if self.mode == "max":
            # _output = np.stack([hz.max(axis=(-1, -2)) for hz in _pooled], axis=2)
            _output = _windows.max(axis=(-1, -2))
        elif self.mode == "min":
            # _output = np.stack([hz.min(axis=(-1, -2)) for hz in _pooled], axis=2)
            _output = _windows.min(axis=(-1, -2))
        elif self.mode == "avg":
            # _output = np.stack([hz.mean(axis=(-1, -2)) for hz in _pooled], axis=2)
            _output = _windows.mean(axis=(-1, -2))

        if self.return_indices and self.mode in ["max", "min"]:
            self.indices = _output.repeat(self.stride, axis=-1).repeat(self.stride, axis=-2)

        return _output

    def backward(self, _input, _gradPrev):
        if not self.return_indices: # must be True to enable backpropagation
            raise Exception("Module not equipped to handle backwards propagation")
        if _gradPrev.shape[-1] != _gradPrev.shape[-2]:
            raise Exception("Adjoint state has incorrect spatial dimensions")
        if _gradPrev.shape[-1] != self.h:
            raise Exception("Adjoint state does not have matching dimensions with internals of the module")

        y = _gradPrev.repeat(self.stride, axis=-1).repeat(self.stride, axis=-2)

        # if stride > kernel_size, we have to zero out the elements from the gradient
        # which were not included in the kernel but are included in the stride
        if self.kernel_size < self.stride:
            mask = [] # assumes that _input spatial dimensions are squares
            diff = self.stride - self.kernel_size
            for i in range(diff):
                mask.append(np.arange(self.kernel_size+i, _input.shape[-1], self.stride))
            mask = np.concatenate(mask).astype(int).tolist()
            mask = np.ix_(mask, mask)

            y[:, :, mask[0], :] = 0 # zero out rows
            y[:, :, :, mask[1]] = 0 # zero out columns

        if self.mode == "max" or self.mode == "min":
            # apply a mask such that only the maximum or minimum value of each kernel
            # has the gradient passed backwards to it
            _gradCurr = np.equal(_input, self.indices).astype(int) * y
        elif self.mode == "avg":
            # scale gradient down by 1 / (H * W) - each element in the kernel gets
            # an equal proportion of the gradient
            _gradCurr = y / (self.kernel_size ** 2)

        if _gradCurr.shape != _input.shape:
            raise Exception("Current gradient does not match dimensions of input vector")

        return _gradCurr

    def params(self):
        return None, None

    def name(self):
        return self.mode.capitalize() + "Pool2d Layer"


def test_pool2d():
    test = Pooling2d(kernel_size=2, stride=3, mode="avg")
    _I = np.random.randint(1, 12, size=(100, 3, 6, 6))
    _G = np.random.randn(100, 3, 2, 2)
    _output = test.forward(_I)
    _gradOutput = test.backward(_I, _G)


if __name__ == '__main__':
    test_pool2d()
