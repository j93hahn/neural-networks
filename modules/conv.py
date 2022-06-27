from .module import Module
import numpy as np


class Conv2d(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, _input):
        ...

    def backward(self, _gradPrev):
        ...

    def update_params(self, alpha):
        ...

    def type(self):
        return "Conv2d Layer"


class MaxPool(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, _input):
        ...

    def backward(self, _gradPrev):
        ...

    def update_params(self, alpha):
        ...

    def type(self):
        return "MaxPooling Layer"
