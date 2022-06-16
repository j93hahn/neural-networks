import numpy as np
from pudb import set_trace


########################
#  LAYER CONSTRUCTION  #
########################


class BaseLayer():
    def __init__(self) -> None:
        self.train = True
        return

    def forward(self, _input):
        pass

    def backward(self):
        pass

    def parameters(self):
        pass

    def training(self):
        self.train = True

    def evaluating(self):
        self.train = False


class FullyConnectedLayer(BaseLayer):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        # He initialization - optimized for ReLU activation
        self.weights = np.random.randn(output_dim, input_dim) * np.sqrt(2/input_dim)
        self.biases = np.zeros((output_dim, 1))

    def forward(self, _input):
        self._input = _input
        self._output = np.dot(self.weights, self._input) + self.biases
        return self._output

    def backward(self):
        ...

    def parameters(self):
        return

    def type(self):
        return "Fully Connected Layer"


class Sequential(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def size(self):
        return len(self.layers)

    def components(self):
        for i in range(self.size()):
            print(self.layers[i].type())

    def forward(self, _input):
        self._inputs = [_input]
        for i in range(self.size()):
            self._inputs.append(self.layers[i].forward(self._inputs[i]))
        self._output = self._inputs[-1]
        return self._output

    def backward(self, input):
        ...

    def parameters(self):
        return

    def training(self):
        BaseLayer.training(self)
        for layer in self.layers:
            layer.training()

    def evaluating(self):
        BaseLayer.evaluating(self)
        for layer in self.layers:
            layer.evaluating()


##########################
#  ACTIVATION FUNCTIONS  #
##########################


class ReLU(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(self, _input):
        self._input = _input
        self._output = np.maximum(0, self._input)
        return self._output

    def backward(self):
        pass

    def parameters(self):
        return None, None

    def type(self):
        return "ReLU Activation"


class Sigmoid(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(self, _input):
        self._input = _input
        self._output = 1. / (1 + np.exp(-self._input))
        return self._output

    def backward(self):
        pass

    def parameters(self):
        return None, None

    def type(self):
        return "Sigmoid Activation"


class SoftMax(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(self, _input):
        self._input = _input
        self._output = np.exp(self._input) / np.sum(np.exp(self._input))
        return np.argmax(self._output)

    def backward(self):
        pass

    def parameters(self):
        return None, None

    def type(self):
        return "SoftMax Activation"


model = Sequential()
model.add(FullyConnectedLayer(784, 16))
model.add(ReLU())
model.add(FullyConnectedLayer(16, 16))
model.add(ReLU())
model.add(FullyConnectedLayer(16, 10))
model.add(SoftMax())


model.forward(np.random.randn(49, 16).reshape(784, 1)) # works properly
#model.backward() # does not work properly
set_trace()


def cost(x, y):
    return np.sum(np.square(x - y))
