from .module import Module
import numpy as np


class Conv2d(Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        # He initialization
        self.weights = np.random.randn(output_dim, input_dim) * np.sqrt(2/input_dim)
        self.biases = np.zeros((output_dim, 1))

        # Gradient arrays for parameters
        self.gradWeights = np.zeros_like(self.weights)
        self.gradBiases = np.zeros_like(self.biases)

    def forward(self, _input):
        ...

    def type():
        return "Conv2d Layer"


class MaxPool(Module):
    def __init__(self) -> None:
        super().__init__()