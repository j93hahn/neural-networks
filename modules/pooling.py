from module import Module
import numpy as np


"""
For all pooling layers, input should be (N x C x H x W)

Key Asumptions:
1) H == W, i.e. _input.shape[2] == _input.shape[3]
2) kernel_size and stride are both integers but do not need to be same value
3) kernel_size <= stride
4) No padding - the pooling can only happen over the given input image plane
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

        h = int(_input.shape[2] / self.stride) # these determine output shape
        grids = [np.ix_(np.arange(self.stride*i, self.stride*i+self.kernel_size),
                        np.arange(self.stride*i, self.stride*i+self.kernel_size)) for i in range(h)]
        _pooled = [_input[:, :, grids[i][0], [grids[j][1] for j in range(h)]] for i in range(h)]

        if self.mode == "Max":
            _output = np.stack([hz.max(axis=(-1, -2)) for hz in _pooled], axis=2)
        elif self.mode == "Min":
            _output = np.stack([hz.min(axis=(-1, -2)) for hz in _pooled], axis=2)
        elif self.mode == "Avg":
            _output = np.stack([hz.mean(axis=(-1, -2)) for hz in _pooled], axis=2)

        if self.return_indices:
            if self.kernel_size == self.stride:
                self.indices = _output.repeat(self.kernel_size, axis=-1).repeat(self.kernel_size, axis=-2)
            else:
                breakpoint()
                # the key to solve this is using np.insert()
                # x = np.insert(np.insert(self.indices, obj=(2, 4), values=0, axis=-1), obj=(2, 4), values=0, axis=-2)
                self.indices = _output.repeat(self.kernel_size, axis=-1).repeat(self.kernel_size, axis=-2)

        return _output

    def backward(self, _input, _gradPrev):
        if not self.return_indices: # must be True to enable backpropagation
            raise Exception("Module not equipped to handle backwards propagation")

        if self.kernel_size == self.stride:
            y = _gradPrev.repeat(self.kernel_size, axis=-1).repeat(self.kernel_size, axis=-2)
        else:
            y = _gradPrev.repeat(self.kernel_size, axis=-1).repeat(self.kernel_size, axis=-2)

        if self.mode == "Max" or self.mode == "Min":
            _gradCurr = np.zeros_like(_input)
            mask = np.equal(_input, self.indices).astype(int) * y # zero-out non-important elements
            _gradCurr[:, :, :mask.shape[2], :mask.shape[3]] = mask
        else: # average pooling
            _gradCurr = _input * y / (self.kernel_size ** 2) # scale gradient down by 1 / (N^2)

        return _gradCurr

    def params(self):
        return None, None

    def name(self):
        return self.mode + "Pool2d Layer"


def test_pool2d():
    test = Pooling2d(kernel_size=2, stride=3, mode="Avg")
    _I = np.random.randint(1, 12, size=(100, 3, 6, 6))
    _G = np.random.randn(100, 3, 2, 2)
    _output = test.forward(_I)
    assert _output.shape == _G.shape
    breakpoint()
    _gradOutput = test.backward(_I, _G)
    assert _gradOutput.shape == _I.shape


if __name__ == '__main__':
    test_pool2d()
