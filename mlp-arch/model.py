import numpy as np
from pudb import set_trace


# https://brilliant.org/wiki/backpropagation/


##########################
##  MODEL CONSTRUCTION  ##
##########################


class Base():
    def __init__(self) -> None:
        self.train = True
        return

    def forward(self, _input):
        pass

    def backprop(self, _input, _gradPrev):
        pass
        """
        _gradPrev is the gradient/delta of the previous layer in the Sequential
            model when applying backpropagation

        return self._gradCurr multiplies the gradient of the current layer and
            passes it to the next layer in the sequence
        """

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

    def backprop(self, _input, _gradPrev):
        self._gradCurr = np.dot(_gradPrev, self.weights.T)
        return self._gradCurr

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

    def backprop(self, _input, _gradPrev):
        ...

    def inputGradient(self, _input):
        # similar logic to forward() -- run through all the layers
        for i in reversed(range(self.size())):
            ...

    def train(self):
        Base.train(self)
        for layer in self.layers:
            layer.train()

    def eval(self):
        Base.eval(self)
        for layer in self.layers:
            layer.eval()


############################
##  ACTIVATION FUNCTIONS  ##
############################


class ReLU(Base):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(self, _input):
        self._input = _input
        self._output = np.maximum(0, self._input)
        return self._output

    def backprop(self, _input, _gradPrev):
        # apply derivative to the input vector space
        self._mask = self._input > 0
        self._gradCurr = _gradPrev * self._mask
        return self._gradCurr

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

    def backprop(self, _input, _gradPrev):
        # calculate derivative on output vector space
        self._mask = self._output * (1 - self._output)
        self._gradCurr = _gradPrev * self._mask
        return self._gradCurr

    def type(self):
        return "Sigmoid Activation"


#######################
##  ERROR FUNCTIONS  ##
#######################


class SoftMaxLoss():
    def __init__(self) -> None:
        return

    def forward(self, _input, _labels):
        # compute softmax activation first
        self._softmax = np.exp(_input) / np.sum(np.exp(_input))
        self._log = np.log(self._softmax)
        self._loss = -np.mean(self._log * _labels, axis=0)
        #self._loss = np.mean(self._loglikelihood)
        return self._loss # a single scalar representing cross entropy loss

    def backprop(self, _input, _labels):
        self._gradCurr
        return self._gradCurr

    def type(self):
        return "SoftMaxLoss"


class MeanSquaredLoss():
    def __init__(self) -> None:
        return

    def forward(self, _input, _labels):
        self._softmax = np.exp(_input) / np.sum(np.exp(_input))
        # eventually, axis=1 if we account for training batches - 500x10x1
        self._loss = np.mean(np.square(self._softmax - _labels), axis=0)
        self._output = self._loss / 2 # may have to np.squeeze here for training batches
        return self._output

    def backprop(self, _input, _labels):
        pass

    def type(self):
        return "MeanSquaredLoss"


model = Sequential()
model.add(Linear(784, 16))
model.add(ReLU())
model.add(Linear(16, 16))
model.add(ReLU())
model.add(Linear(16, 10))

set_trace()
model.forward(np.random.randn(49, 16).reshape(784, 1)) # works properly
#model.backprop() # does not work properly
# model.paramGradient()


# loss = MeanSquaredLoss()
loss = SoftMaxLoss()
prediction = np.array([[0.1], [0.1], [0.1], [0.7]])
actual = np.array([[0], [0], [0], [1]])
set_trace()
value = loss.forward(prediction, actual)
print(value)
