import numpy as np
from pudb import set_trace


##########################
##  MODEL CONSTRUCTION  ##
##########################


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

    def backward(self, _gradPrev):
        # compute weight gradients here
        self.gradWeights.fill(0)
        self.gradBiases.fill(0)

        self.gradWeights = np.dot(self._input, _gradPrev.T)
        self.gradBiases = np.mean(_gradPrev, axis=0).T

        # pass gradient to next layer in backward propagation
        self._gradCurr = np.dot(self.weights.T, _gradPrev)
        return self._gradCurr

    def update_params(self, alpha):
        if self.train:
            self.weights += (alpha * self.gradWeights.T * -1)
            self.biases += (alpha * self.gradBiases * -1)

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

    def backward(self, _gradPrev):
        self._gradPrevArray = [0] * (self.size() + 1)
        self._gradPrevArray[self.size()] = _gradPrev
        for i in reversed(range(self.size())):
            self._gradPrevArray[i] = self.layers[i].backward(self._gradPrevArray[i + 1])

    def update_params(self, alpha_scheduler): # update_params parameters here
        alpha = 0.01 # eventually, implement a scheduler
        for i in range(self.size()):
            self.layers[i].update_params(alpha)

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

    def backward(self, _gradPrev):
        # input and output vectors have same dimension
        self._derivative = self._input > 0
        # self._gradCurr = _gradPrev * self._mask
        self._gradCurr = np.diag(np.squeeze(self._derivative)).dot(_gradPrev)
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

    def backward(self, _gradPrev):
        # calculate derivative on output vector space
        self._mask = self._output * (1 - self._output)
        self._gradCurr = _gradPrev * self._mask
        return self._gradCurr

    def type(self):
        return "Sigmoid Activation"


class SoftMax(Base):
    def __init__(self) -> None:
        super().__init__()
        return

    def forward(self, _input):
        self._input = _input
        self._output = np.exp(self._input) / np.sum(np.exp(self._input))
        return self._output

    def backward(self, _gradPrev):
        # if i == j, return the derivative, else 0
        self._derivative = self._output * (1 - self._output)
        self._gradCurr = np.diag(np.squeeze(self._derivative)).dot(_gradPrev)
        return self._gradCurr

    def type(self):
        return "SoftMax Activation"


#######################
##  ERROR FUNCTIONS  ##
#######################


class CrossEntropyLoss():
    def __init__(self) -> None:
        return

    def loss(self, _input, _labels):
        self._input = _input
        self._log = np.log(self._input)
        self._loss = -np.sum(self._log * _labels)
        return self._loss

    def backward(self, _input, _labels):
        self._grad = _labels - _input
        return self._grad

    def type(self):
        return "CrossEntropyLoss"


class MeanSquaredLoss():
    def __init__(self) -> None:
        return

    def forward(self, _input, _labels):
        self._softmax = np.exp(_input) / np.sum(np.exp(_input))
        # eventually, axis=1 if we account for training batches - 500x10x1
        self._loss = np.mean(np.square(self._softmax - _labels), axis=0)
        self._output = self._loss / 2 # may have to np.squeeze here for training batches
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
model.add(SoftMax())

#set_trace()
input = np.random.randn(49, 16).reshape(784, 1)
predict = model.forward(input) # works properly
actual = np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]])

loss = CrossEntropyLoss()
error = loss.backward(predict, actual)

set_trace()
model.backward(error)

set_trace()
model.update_params(0.01)
