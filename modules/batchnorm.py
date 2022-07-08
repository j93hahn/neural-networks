from .module import Module
import numpy as np


# requires mini-batch gradient descent
class BatchNorm(Module):
    def __init__(self, batchsize, input_dim) -> None:
        super().__init__()
        self._batchsize = batchsize # K-value
        self._norm = (1 / self._batchsize)
        self._n = input_dim # same size as output_dim
        self._G = np.random.uniform(0, 1, size=(self._n, 1))
        self._B = np.zeros_like(self._G)
        self._eps = 1e-5

    def forward(self, _input):
        self._input = _input # n x K
        self._mean = self._norm * np.sum(self._input, axis=-1)[:, np.newaxis] # n x 1
        self._std = np.sqrt(self._norm * np.sum(np.square(self._input - self._mean), axis=-1))[:, np.newaxis] # n x 1
        self._Zbar = (self._input - self._mean) / (self._std + self._eps)
        self._Zhat = (self._G * self._Zbar) + self._B # output value

    def backward(self, _gradPrev):
        # schematically, dL/dB = dL/dZ_hat * dZ_hat/dB
        # but, dZhat/dB = 1, so it's multiplied out and we get dL/dZ_hat (_gradPrev)
        self.l_dB = _gradPrev # not sure about this calc
        self.l_dG = np.dot(_gradPrev, self._Zbar) # not sure about this calc

        """
        dL/dZ_ij = sum (from k = 1 to K) of dL/dZhat_ik * G_i/(K*self._std_i)
        * ()

        very complicated formulation
        """
        self._gradCurr = ...
        return self._gradCurr

    def params(self):
        return [], []

    def update_params(self, time):
        alpha = 0.01
        self._B -= alpha * self.l_dB
        self._G -= alpha * self.l_dG

    def type(self):
        return "Batch Normalization Layer"


# set_trace()
test = BatchNorm(50, 150)


x = np.ones((150, 50))
#set_trace()
test.forward(x)
