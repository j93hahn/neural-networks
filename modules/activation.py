from .module import Module
import numpy as np
from pudb import set_trace


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(self, _input):
        # _input has shape NxF, self._mask has shape NxF
        self._mask = _input > 0
        return np.maximum(0, _input)

    def backward(self, _input, _gradPrev):
        # input and output vectors have same dimension
        return _gradPrev * self._mask

    def params(self):
        return None, None

    def name(self):
        return "ReLU Activation"


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(self, _input):
        _output = 1. / (1 + np.exp(_input))
        self._mask = _output * (1 - _output)
        return _output

    def backward(self, _gradPrev):
        # calculate derivative on output vector space
        return _gradPrev * self._mask

    def name(self):
        return "Sigmoid Activation"


class SoftMax(Module):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(self, _input):
        # to prevent overflow, subtract an offset d from each input
        # d is set to be maximum of x_i, so 1 for MNIST
        # you might wonder, what about underflow? it's okay, the gradient will be set to 0 anyways
        #set_trace()
        num = np.exp(_input - _input.max())
        denom = np.sum(num)
        self._output = num / denom
        return self._output
        # self._mask = _output * (1 - _output)

    def backward(self, _input, _gradPrev):
        #set_trace()
        _ij = np.diag(np.squeeze(self._output)) # n by n

        _ijnot = np.tile(self._output, 10) * np.tile(self._output, 10).T# n by n

        return (_ij - _ijnot).T.dot(_gradPrev) # results in n by 1
        # _gradPrev has shape 10
        # return _gradPrev * self._mask
        # return np.diag(np.squeeze(self._mask)).dot(_gradPrev)

    def params(self):
        return None, None

    def name(self):
        return "SoftMax Activation"
