import numpy as np
from pudb import set_trace


# https://brilliant.org/wiki/backpropagation/


########################
#  MODEL CONSTRUCTION  #
########################


class Base():
    def __init__(self) -> None:
        self.train = True
        return

    def forward(self, _input):
        pass

    def inputGradient(self, _input):
        pass

    def paramGradient(self):
        pass

    def train(self):
        self.train = True

    def eval(self):
        self.train = False


class Linear(Base):
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
        return "Linear Layer"


class Sequential(Base):
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

    def train(self):
        Base.train(self)
        for layer in self.layers:
            layer.train()

    def eval(self):
        Base.eval(self)
        for layer in self.layers:
            layer.eval()


##########################
#  ACTIVATION FUNCTIONS  #
##########################


class ReLU(Base):
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


class Sigmoid(Base):
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


class CrossEntropyLoss():
    def __init__(self) -> None:
        return

    def forward(self, _input, _labels):
        # compute softmax activation first
        self._softmax = np.exp(_input) / np.sum(np.exp(_input))
        self._loss = -np.sum(np.log(self._softmax) * _labels)
        self._output = self._loss # to normalize, divide by _input.size
        return self._output # a single scalar representing cross entropy loss

    def backward(self, _input, _labels):
        # not sure what derivative is
        pass

    def type(self):
        return "CrossEntropyLoss"


class MeanSquaredLoss():
    def __init__(self) -> None:
        return

    def forward(self, _input, _labels):
        self._softmax = np.exp(_input) / np.sum(np.exp(_input))
        self._loss = np.sum(np.square(self._softmax - _labels))
        self._output = self._loss # to normalize, divide by _input.size
        return self._output

    def backward(self, _input, _labels):
        pass

    def type(self):
        return "MeanSquaredLoss"


model = Sequential()
model.add(Linear(784, 16))
model.add(ReLU())
model.add(Linear(16, 16))
model.add(ReLU())
model.add(Linear(16, 10))


model.forward(np.random.randn(49, 16).reshape(784, 1)) # works properly
#model.backprop() # does not work properly
# model.paramGradient()
set_trace()
