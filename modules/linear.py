from .module import Module
import numpy as np
from pudb import set_trace


class Linear(Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        # Gaussian distribution initialization
        self.weights = np.random.normal(0, 1 / in_features, (out_features, in_features))
        self.biases = np.random.normal(0, 1, out_features)

        # PyTorch implementation
        #self.weights = np.random.normal(-np.sqrt(1/in_features), np.sqrt(1/in_features), (out_features, in_features))
        #self.biases = np.random.normal(-np.sqrt(1/in_features), np.sqrt(1/in_features), out_features)

        # Kaiming He initialization
        #self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        #self.biases = np.zeros((1, output_dim))

        # Fan in and fan out from pytorch - look at derivation/source code
        # look at Kaiming He initialization and why it's used
        # take a look at SIREN arxiv paper too

        # for gradient descent
        self.gradWeights = np.zeros_like(self.weights)
        self.gradBiases = np.zeros_like(self.biases)

        # for Adam optimization
        #self._alpha = 0.01
        #self._b1 = 0.9
        #self._b2 = 0.999
        #self._eps = 1e-8
        # m = mean, v = variance
        #self.m_dw, self.v_dw = 0, 0 # gradients w.r.t. to the weights
        #self.m_db, self.v_db = 0, 0 # gradients w.r.t. to the biases

    def forward(self, _input):
        # input is NxF where F is number of in features, and N is number of data samples
        return np.dot(_input, self.weights.T) + self.biases

    def backward(self, _input, _gradPrev):
        # compute gradients here
        """
        _gradPrev has shape N x out_features

        Assume dL/dY has already been computed, a.k.a. _gradPrev
        dL/dX = dL/dY * W.T
        dL/dW = _input.T * dL/dY
        dL/dB =
        """
        #set_trace()
        _gradCurr = np.dot(_gradPrev, self.weights)
        self.gradWeights = _gradPrev.T.dot(_input) / _input.shape[0]
        self.gradBiases = np.sum(_gradPrev, axis=0) / _input.shape[0]

        #set_trace()
        alpha=0.1
        self.weights -= (alpha * self.gradWeights)
        self.biases -= (alpha * self.gradBiases)
        #self.gradWeights = np.dot(self._input, _gradPrev.T)
        #self.gradBiases = np.sum(_gradPrev.T, axis=0, keepdims=True)

        # pass gradient to next layer in backward propagation
        return _gradCurr

    def update_params(self, time):
        # gradient descent without Adam optimization
        alpha = 0.1
        #self.weights -= (alpha * self.gradWeights)
        #self.biases -= (alpha * self.gradBiases)

        # gradient descent with Adam optimization
        #self.m_dw = self._b1 * self.m_dw + (1 - self._b1) * self.gradWeights
        #self.m_db = self._b1 * self.m_db + (1 - self._b1) * self.gradBiases
        #self.v_dw = self._b2 * self.v_dw + (1 - self._b2) * np.square(self.gradWeights)
        #self.v_db = self._b2 * self.v_db + (1 - self._b2) * np.square(self.gradBiases)

        #m_dw_hat = self.m_dw / (1 - np.power(self._b1, time + 1))
        #m_db_hat = self.m_db / (1 - np.power(self._b1, time + 1))
        #v_dw_hat = self.v_dw / (1 - np.power(self._b2, time + 1))
        #v_db_hat = self.v_db / (1 - np.power(self._b2, time + 1))

        # set_trace()
        #self.weights -= ((self._alpha * m_dw_hat) / (np.sqrt(v_dw_hat) + self._eps)).T
        #self.biases -= ((self._alpha * m_db_hat) / (np.sqrt(v_db_hat) + self._eps))

    def params(self):
        return [self.weights, self.biases], [self.gradWeights, self.gradBiases]

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
