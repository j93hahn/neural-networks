import numpy as np
from typing import List


class Optimizer(object):
    def __init__(self, params: List[list]):
        self.params = params[0]
        self.gradParams = params[1]

    def zero_grad(self):
        # fill the gradient arrays with 0s
        #breakpoint()
        for i in range(len(self.gradParams)):
            for j in range(len(self.gradParams[i])):
                self.gradParams[i][j].fill(0)

    def step(self):
        # conduct optimization step - varies based on the optimizer
        pass


class Standard(Optimizer):
    def __init__(self, params: List[list]):
        super().__init__(params)

    def step(self):
        #breakpoint()
        alpha=0.01
        for i in range(len(self.params)):
            for j in range(len(self.params[i])):
                self.params[i][j] -= alpha * self.gradParams[i][j]
