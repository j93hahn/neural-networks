from .module import Module
import numpy as np


class Linear(Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        # Gaussian distribution initialization
        self.weights = np.random.normal(0, 1 / input_dim, (output_dim, input_dim))
        self.biases = np.random.normal(0, 1, (output_dim, 1))

        # Gradient arrays for parameters
        self.gradWeights = np.zeros_like(self.weights)
        self.gradBiases = np.zeros_like(self.biases)

    def forward(self, _input):
        self._input = _input
        self._output = np.dot(self.weights, self._input) + self.biases
        return self._output

    def backward(self, _gradPrev):
        # compute weight gradients here
        self.gradWeights.fill(0)
        self.gradBiases.fill(0)

        self.gradWeights = np.dot(self._input, _gradPrev.T)
        self.gradBiases = np.mean(_gradPrev, axis=0)

        # pass gradient to next layer in backward propagation
        self._gradCurr = np.dot(self.weights.T, _gradPrev)
        return self._gradCurr

    def update_params(self, alpha):
        if self.train:
            self.weights += (alpha * self.gradWeights.T * -1)
            self.biases += (alpha * self.gradBiases.T * -1)

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

    def backward(self, _gradPrev):
        # scale the backwards pass by the same amount
        self._gradCurr = _gradPrev
        self._gradCurr *= self._distribution
        return self._gradCurr

    def type(self):
        return "Dropout Layer"
