import numpy as np
from pudb import set_trace


class Base():
    def __init__(self) -> None:
        self.train = True
        return

    def forward(self, _input):
        pass

    def backward(self, _gradPrev):
        pass
        """
        _gradPrev is the gradient/delta of the previous layer in the Sequential
            model when applying backwardagation

        return self._gradCurr multiplies the gradient of the current layer and
            passes it to the next layer in the sequence
        """

    def update_params(self, alpha):
        pass

    def train(self):
        self.train = True

    def eval(self):
        self.train = False


class Conv2d(Base):
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


class MaxPool(Base):
    def __init__(self) -> None:
        super().__init__()


