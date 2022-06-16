import numpy as np

"""
Steps for building a neural network model:
1. https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
2. SGD: https://www.youtube.com/watch?v=k3AiUhwHQ28
"""

class Layer():
    def __init__(self, input_dim, output_dim, input=False, output=False) -> None:
        # He initialization - optimized for ReLU activation
        if not input:
            self.weights = np.random.randn(output_dim, input_dim) * np.sqrt(2/input_dim)
            self.biases = np.zeros((output_dim, 1))

    def ReLU(self, a):
        return np.maximum(0, a) # broadcasting 0 to input dimensionality

    # return the layer's output for the given input a
    def forward(self, a):
        result = self.ReLU(np.dot(self.weights, a) + self.biases)
        return result

    def backward(self):
        """
        back propagation looks like this:

        """
        if self.output:
            ...


class Sequential(Layer):
    def __init__(self) -> None:
        Layer.__init__(self)
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def size(self):
        return len(self.layers)

    def forward(self, input):
        pass

    def backward(self, input):
        pass

    def training(self):
        pass

    def evaluate(self):
        pass


class ReLU():
    def __init__(self) -> None:
        pass


def SoftMaxActivation(z):
    return np.exp(z) / np.sum(np.exp(z))

example = Layer(784, 32)