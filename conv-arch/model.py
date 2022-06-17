import sys
sys.path.insert(0, '../mlp-arch/')

import numpy as np
from pudb import set_trace
from model import Base, Sequential, Linear, ReLU


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


# deep residual network class (ResNet)
class ResNet():
    def __init__(self) -> None:
        pass


model = Sequential()
model.add(Linear(784, 32))
model.add(ReLU())
model.add(Linear(32, 16))
#print(model.components())
model.components()