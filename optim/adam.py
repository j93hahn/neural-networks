import numpy as np
from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, a=0.01, betas=[0.9, 0.999], eps=1e-8) -> None:
        super().__init__()
        self._alpha = a
        self._b1 = betas[0]
        self._b2 = betas[1]
        self._eps = eps
        # initialize derivatives of mean and variance w.r.t weights, biases
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0

    def step(self, time):
        # gradient descent via Adam optimization
        self.m_dw = self._b1 * self.m_dw + (1 - self._b1) * self.gradWeights
        self.m_db = self._b1 * self.m_db + (1 - self._b1) * self.gradBiases
        self.v_dw = self._b2 * self.v_dw + (1 - self._b2) * np.square(self.gradWeights)
        self.v_db = self._b2 * self.v_db + (1 - self._b2) * np.square(self.gradBiases)

        m_dw_hat = self.m_dw / (1 - np.power(self._b1, time + 1))
        m_db_hat = self.m_db / (1 - np.power(self._b1, time + 1))
        v_dw_hat = self.v_dw / (1 - np.power(self._b2, time + 1))
        v_db_hat = self.v_db / (1 - np.power(self._b2, time + 1))

        self.weights -= ((self._alpha * m_dw_hat) / (np.sqrt(v_dw_hat) + self._eps)).T
        self.biases -= ((self._alpha * m_db_hat) / (np.sqrt(v_db_hat) + self._eps))
