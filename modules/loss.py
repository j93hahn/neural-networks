import numpy as np
from pudb import set_trace


class CrossEntropyLoss(object):
    def __init__(self) -> None:
        return

    def loss(self, _input, _labels):
        self._log = np.log(_input)
        self._loss = -np.sum(self._log * _labels)
        return self._loss

    def backward(self, _input, _labels):
        #set_trace()
        return _labels - _input

    def type(self):
        return "Cross Entropy Loss"


class MSELoss(object):
    def __init__(self) -> None:
        return

    def forward(self, _input, _labels):
        self._softmax = np.exp(_input) / np.sum(np.exp(_input))
        # eventually, axis=1 if we account for training batches - 500x10x1
        self._loss = np.mean(np.square(self._softmax - _labels), axis=0)
        self._output = self._loss / 2 # may have to np.squeeze here for training batches
        return self._output

    def backward(self, _input, _labels):
        pass

    def type(self):
        return "Mean Squared Error Loss"
