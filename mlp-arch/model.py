import numpy as np
from pudb import set_trace


########################
#  MODEL CONSTRUCTION  #
########################


class BaseLayer():
    def __init__(self) -> None:
        self.train = True
        return

    def forward(self, _input):
        pass

    def inputGradient(self, _input):
        pass

    def paramGradient(self):
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

        # Gradient arrays for parameters
        self.gradWeights = np.zeros_like(self.weights)
        self.gradBiases = np.zeros_like(self.biases)

    def forward(self, _input):
        self._input = _input
        self._output = np.dot(self.weights, self._input) + self.biases
        return self._output

    # compute gradient with respect to _input vector
    def inputGradient(self, _input):
        return

    # return all parameters and their gradients
    def paramGradient(self):
        return self.weights, self.biases, self.gradWeights, self.gradBiases

    def type(self):
        return "Fully Connected Layer"


class Sequential(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        self.layers = []

    def size(self):
        return len(self.layers)

    def components(self):
        for i in range(self.size()):
            print(self.layers[i].type())

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, _input):
        self._inputs = [_input]
        for i in range(self.size()):
            self._inputs.append(self.layers[i].forward(self._inputs[i]))
        self._output = self._inputs[-1]
        return self._output

    def inputGradient(self, _input):
        # similar logic to forward() -- run through all the layers
        for layer in reversed(range(self.size())):
            ...

    def paramGradient(self):
        w = []; b = []; wg = []; wb = []
        for layer in self.layers:
            _w, _b, _wg, _wb = layer.paramGradient()
            # activation layers do no have trainable parameters
            if np.all(_w != None):
                w.append(_w); b.append(_b)
                wg.append(_wg); wb.append(_wb)
        return w, b, wg, wb

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

    def inputGradient(self, _input):
        # derivative is just _input > 0
        pass

    def paramGradient(self):
        return None, None, None, None

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

    def inputGradient(self, _input):
        # not sure what derivative is
        pass

    def paramGradient(self):
        return None, None, None, None

    def type(self):
        return "Sigmoid Activation"


class SoftMax(BaseLayer):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(self, _input):
        self._input = _input
        self._output = np.exp(self._input) / np.sum(np.exp(self._input))
        return self._output
        # np.argmax(self._output) gives the highest probability output

    def inputGradient(self, _input):
        # not sure what derivative is
        pass

    def paramGradient(self):
        return None, None, None, None

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
#model.backprop() # does not work properly
model.paramGradient()
set_trace()


def cost(x, y):
    return np.sum(np.square(x - y))
