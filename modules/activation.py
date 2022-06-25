from .module import Module
import numpy as np


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(self, _input):
        self._input = _input
        self._output = np.maximum(0, self._input)
        return self._output

    def backward(self, _gradPrev):
        # input and output vectors have same dimension
        self._derivative = self._input > 0
        # self._gradCurr = _gradPrev * self._mask
        self._gradCurr = np.diag(np.squeeze(self._derivative)).dot(_gradPrev)
        return self._gradCurr

    def type(self):
        return "ReLU Activation"


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(self, _input):
        self._input = _input
        self._output = 1. / (1 + np.exp(-self._input))
        return self._output

    def backward(self, _gradPrev):
        # calculate derivative on output vector space
        self._mask = self._output * (1 - self._output)
        self._gradCurr = _gradPrev * self._mask
        return self._gradCurr

    def type(self):
        return "Sigmoid Activation"


class SoftMax(Module):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(self, _input):
        self._input = _input
        self._denom = np.sum(np.exp(self._input))
        self._output = np.exp(self._input) / self._denom
        return self._output, self._denom

    def backward(self, _gradPrev):
        # if i == j, return the derivative, else 0
        self._derivative = self._output * (1 - self._output)
        self._gradCurr = np.diag(np.squeeze(self._derivative)).dot(_gradPrev)
        return self._gradCurr

    def type(self):
        return "SoftMax Activation"
