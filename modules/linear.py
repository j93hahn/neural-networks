from .module import Module
import numpy as np
from pudb import set_trace


class Linear(Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        # Gaussian distribution initialization
        self.weights = np.random.normal(0, 1 / input_dim, (output_dim, input_dim))
        self.biases = np.random.normal(0, 1, (output_dim, 1))

        # for gradient descent
        self.gradWeights = np.zeros_like(self.weights)
        self.gradBiases = np.zeros_like(self.biases)

        # for Adam optimization
        self._eta = 0.01
        self._b1 = 0.9
        self._b2 = 0.999
        self._eps = 1e-8
        # m = mean, v = variance
        self.m_dw, self.v_dw = 0, 0 # gradients w.r.t. to the weights
        self.m_db, self.v_db = 0, 0 # gradients w.r.t. to the biases

    def forward(self, _input):
        self._input = _input
        self._output = np.dot(self.weights, self._input) + self.biases
        return self._output

    def backward(self, _gradPrev):
        # compute weight gradients here
        self.gradWeights.fill(0)
        self.gradBiases.fill(0)

        self.gradWeights = np.dot(self._input, _gradPrev.T)
        self.gradBiases = np.mean(_gradPrev, axis=0) # check this logic later

        # pass gradient to next layer in backward propagation
        self._gradCurr = np.dot(self.weights.T, _gradPrev)
        return self._gradCurr

    def update_params(self, time):
        # gradient descent without Adam optimization
        # alpha = 0.01
        # self.weights += (alpha * self.gradWeights.T * -1)
        # self.biases += (alpha * self.gradBiases.T * -1)

        # gradient descent with Adam optimization
        self.m_dw = self._b1 * self.m_dw + (1 - self._b1) * self.gradWeights
        self.m_db = self._b1 * self.m_db + (1 - self._b1) * self.gradBiases
        self.v_dw = self._b2 * self.v_dw + (1 - self._b2) * (self.gradWeights ** 2)
        self.v_db = self._b2 * self.v_db + (1 - self._b2) * (self.gradBiases ** 2)

        m_dw_hat = (self.m_dw / (1 - (self._b1 ** time))).T
        m_db_hat = self.m_db / (1 - (self._b1 ** time))
        v_dw_hat = (self.v_dw / (1 - (self._b2 ** time))).T
        v_db_hat = self.v_db / (1 - (self._b2 ** time))

        # set_trace()
        self.weights -= (self._eta * m_dw_hat) / np.sqrt(self._eps + v_dw_hat)
        self.biases -= (self._eta * m_db_hat) / np.sqrt(self._eps + v_db_hat)

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
