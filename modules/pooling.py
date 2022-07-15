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
    def __init__(self, kernel_size, stride=None, mode="Max", return_indices=True) -> None:
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

        if mode not in ["Max", "Min", "Avg"]:
            raise Exception("Invalid pooling mode specified")
        self.mode = mode

    def forward(self, _input):
        """
        Example Calculation:

        Note: Number of grids N = H / stride
        Each grid should go from 0+stride*n to stride*(n+1) from n is in range(0, N)

        a = np.arange(540).reshape(3, 5, 6, 6)

        grid1 = np.ix_([0, 1], [0, 1])
        grid2 = np.ix_([2, 3], [2, 3])
        grid3 = np.ix_([4, 5], [4, 5])

        a1 = a[:, :, grid1[0], [grid1[1], grid2[1], grid3[1]]].max(axis=(-1, -2))
        a2 = a[:, :, grid2[0], [grid1[1], grid2[1], grid3[1]]].max(axis=(-1, -2))
        a3 = a[:, :, grid3[0], [grid1[1], grid2[1], grid3[1]]].max(axis=(-1, -2))

        output = np.stack((a1, a2, a3), axis=2)
        """
        if _input.shape[2] % self.stride:
            raise Exception("Invalid stride shape for input size shape")
        if _input.shape[2] != _input.shape[3]:
            raise Exception("Input spatial dimension axes do not match")

        self.h = int(_input.shape[2] / self.stride) # these determine output shape
        grids = [np.ix_(np.arange(self.stride*i, self.stride*i+self.kernel_size),
                        np.arange(self.stride*i, self.stride*i+self.kernel_size)) for i in range(self.h)]
        _pooled = [_input[:, :, grids[i][0], [grids[j][1] for j in range(self.h)]] for i in range(self.h)]

        if self.mode == "Max":
            _output = np.stack([hz.max(axis=(-1, -2)) for hz in _pooled], axis=2)
        elif self.mode == "Min":
            _output = np.stack([hz.min(axis=(-1, -2)) for hz in _pooled], axis=2)
        elif self.mode == "Avg":
            _output = np.stack([hz.mean(axis=(-1, -2)) for hz in _pooled], axis=2)

        if self.return_indices and self.mode in ["Max", "Min"]:
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
        if self.kernel_size < self.stride:
            _pads = []
            diff = self.stride - self.kernel_size
            for i in range(diff):
                _pads.append(np.arange(self.kernel_size+i, _input.shape[2], self.stride))
            _pads = np.concatenate(_pads).astype(int).tolist()
            _pads = np.ix_(_pads, _pads)

            y[..., _pads[0], :] = 0
            y[..., :, _pads[1]] = 0

        if self.mode == "Max" or self.mode == "Min":
            _gradCurr = np.equal(_input, self.indices).astype(int) * y
        else: # average pooling
            _gradCurr = _input * y / (self.kernel_size ** 2) # scale gradient down by 1 / (N^2)

        if _gradCurr.shape != _input.shape:
            raise Exception("Current gradient does not match dimensions of input vector")

        return _gradCurr

    def params(self):
        return None, None

    def name(self):
        return self.mode + "Pool2d Layer"


def test_pool2d():
    test = Pooling2d(kernel_size=5, stride=6, mode="Avg")
    _I = np.random.randint(1, 12, size=(100, 3, 30, 30))
    _G = np.random.randn(100, 3, 5, 5)
    _output = test.forward(_I)
    _gradOutput = test.backward(_I, _G)


if __name__ == '__main__':
    test_pool2d()
