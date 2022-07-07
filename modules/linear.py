from .module import Module
import numpy as np
from pudb import set_trace


class Linear(Module):
    def __init__(self, in_features, out_features, init_method="Gaussian") -> None:
        super(Linear, self).__init__()

        if init_method == "Gaussian":
            # Gaussian initialization
            self.weights = np.random.normal(0, 1 / in_features, (out_features, in_features))
            self.biases = np.random.normal(0, 1, out_features)
        elif init_method == "He":
            # He initialization
            self.weights = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
            self.biases = np.zeros((1, out_features))
        elif init_method == "PyTorch":
            # PyTorch initialization
            self.weights = np.random.normal(-np.sqrt(1/in_features), np.sqrt(1/in_features), (out_features, in_features))
            self.biases = np.random.normal(-np.sqrt(1/in_features), np.sqrt(1/in_features), out_features)
        else:
            raise Exception("Invalid Initialization technique")

        # Gradient descent initialization
        self.gradWeights = np.zeros_like(self.weights)
        self.gradBiases = np.zeros_like(self.biases)

    def forward(self, _input):
        # input is NxF where F is number of in features, and N is number of data samples
        return np.dot(_input, self.weights.T) + self.biases

    def backward(self, _input, _gradPrev):
        """
        _gradPrev has shape N x out_features

        Assume dL/dY has already been computed, a.k.a. _gradPrev
        dL/dX = dL/dY * W.T
        dL/dW = _input.T * dL/dY
        dL/dB = sum(dL/dY, axis=0)
        """
        #self.gradWeights.fill(0)
        _gradCurr = np.dot(_gradPrev, self.weights)
        self.gradWeights = _gradPrev.T.dot(_input) / _input.shape[0]
        self.gradBiases = np.sum(_gradPrev, axis=0) / _input.shape[0]

        alpha=0.01
        self.weights -= (alpha * self.gradWeights)
        self.biases -= (alpha * self.gradBiases)

        return _gradCurr

    def params(self):
        return self.weights, self.biases, self.gradWeights, self.gradBiases

    def type(self):
        return "Linear Layer"


class Dropout(Module):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self._p = p

    def forward(self, _input):
        self._output = _input
        self._distribution = np.random.binomial(n=1, p=1-self._p, size=_input.shape)
        self._output *= self._distribution
        return self._output

    def backward(self, _input, _gradPrev):
        # scale the backwards pass by the same amount
        self._gradCurr = _gradPrev
        self._gradCurr *= self._distribution
        return self._gradCurr

    def params(self):
        return None, None

    def type(self):
        return "Dropout Layer"
