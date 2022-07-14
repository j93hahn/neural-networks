from module import Module
import numpy as np


"""
For all pooling layers, input should be (N x C x H x W)

Key Asumptions:
1) H == W, i.e. _input.shape[2] == _input.shape[3]
2) kernel_size and stride are both integers but do not need to be same value
3) No padding - the pooling can only happen over the given input image plane
"""
class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, return_indices=True) -> None:
        super().__init__()
        self.kernel_size = kernel_size # input must be an integer
        self.stride = kernel_size if stride == None else stride
        if type(self.kernel_size) != type(self.stride):
            raise Exception("Kernel size and stride input types do not match")
        self.return_indices = return_indices # if true, return index of max value, necessary for MaxUnpool2d

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
        breakpoint()
        if isinstance(self.stride, int):
            if _input.shape[2] % self.stride:
                raise Exception("Invalid kernel size for input size shape")
            if self.stride < self.kernel_size:
                raise Exception("Kernel size is larger than the stride")
            h = int(_input.shape[2] / self.stride) # these determine output shape
            grids = [np.ix_(np.arange(self.stride*i, self.stride*i+self.kernel_size),
                            np.arange(self.stride*i, self.stride*i+self.kernel_size)) for i in range(h)]
            _pooled = [_input[:, :, grids[i][0], [grids[j][1] for j in range(h)]] for i in range(h)]
            pooled = [element.max(axis=(-1, -2)) for element in _pooled]
            output = np.stack(pooled, axis=2)
            if self.return_indices:
                # https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html
                # store indices of maximum values. in backward pass, set all other values as 0
                _indices = [element.reshape(element.shape[0], element.shape[1], element.shape[2], -1).argmax(-1) for element in _pooled]
                self._index = np.stack(_indices, axis=2)
            return output
        else:
            raise Exception("Invalid stride input type")

    def backward(self, _input, _gradPrev):
        if not self.return_indices: # must be True to enable backpropagation
            raise Exception("Module not equipped to handle backwards propagation")
        ...

    def params(self):
        pass

    def name(self):
        return "Max Pool 2d Layer"


class MinPool2d(Module):
    def __init__(self, stride=2) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, _input):
        ...

    def backward(self, _input, _gradPrev):
        ...

    def params(self):
        pass

    def name(self):
        return "Min Pool 2d Layer"


class AvgPool2d(Module):
    def __init__(self, stride=2) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, _input):
        ...

    def backward(self, _input, _gradPrev):
        ...

    def params(self):
        pass

    def name(self):
        return "Average Pool 2d Layer"


def main():
    test = MaxPool2d(kernel_size=2)
    _input = np.arange(540).reshape(3, 5, 6, 6)
    test.forward(_input)


if __name__ == '__main__':
    main()
