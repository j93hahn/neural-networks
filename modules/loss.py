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

    def loss(self, _input, _labels):
        #self._softmax = np.exp(_input) / np.sum(np.exp(_input))
        # eventually, axis=1 if we account for training batches - 500x10x1
        _loss = np.mean(np.square(_input - _input.max() - _labels))
        #self._output = self._loss / 2 # may have to np.squeeze here for training batches
        return _loss

    def backward(self, _input, _labels):
        return (_input - _input.max() - _labels) * -2 / _input.shape[0]

    def type(self):
        return "Mean Squared Error Loss"


class SoftMaxLoss(object):
    def __init__(self) -> None:
        """
        Combines softmax activation with cross entropy loss (negative log likelihood
        loss - NLLLoss) in order to provide ease with gradient descent

        Input vector size: N x C
        Labels size: N x C, one-hot encoded
        N = number of samples, C = number of classes (image classification task)

        Forward Steps:
        1. Compute softmax of input vector along axis = 1
        2. Calculate log likelihood of input vector
        3. Generate sums for each sample and return averaged log loss

        Backward Steps:
        1. The derivative is simply -1 * (_labels - exp(self._loglikelihood))
        2. The math works out very nicely, it's a very simple derivation
        Shape of tensor: N x C

        Short Note: We need the [:, np.newaxis] to account for the dimensionality
        of the input space
        """
        pass

    def forward(self, _input, _labels):
        _softmax = _input - _input.max(axis=1)[:, np.newaxis]
        self._loglikelihood = _softmax - np.log(np.sum(np.exp(_softmax), axis=1))[:, np.newaxis]
        _sums = np.sum(-self._loglikelihood * _labels, axis=1) # generate a N x 1 vector
        return np.mean(_sums) # generate averaged log loss for all N samples

    def backward(self, _labels):
        return (np.exp(self._loglikelihood) - _labels) # / _labels.shape[0]

    def type(self):
        return "Softmax Log Loss"
