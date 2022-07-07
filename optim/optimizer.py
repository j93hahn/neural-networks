import numpy as np


class Optimizer(object):
    def __init__(self, params):
        self.params = params[0]
        self.gradParams = ...

    def zero_grad(self):
        # fill the gradient arrays with 0s
        pass

    def step(self):
        # conduct optimization step
        pass