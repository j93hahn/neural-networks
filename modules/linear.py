from .module import Module
import numpy as np


class Linear(Module):
    def __init__(self, in_features, out_features, init_method="Gaussian") -> None:
        super(Linear, self).__init__()

        if init_method == "Zero":
            # Zeros initialization
            self.weights = np.zeros((out_features, in_features))
            self.biases = np.zeros(out_features)
        elif init_method == "Random":
            # Random distribution
            self.weights = np.random.randn(out_features, in_features)
            self.biases = np.random.randn(out_features)
        elif init_method == "Gaussian":
            # Gaussian normal distribution
            self.weights = np.random.normal(0, 1 / in_features, (out_features, in_features))
            self.biases = np.random.normal(0, 1, out_features)
        elif init_method == "He":
            # He initialization - https://arxiv.org/pdf/1502.01852.pdf
            self.weights = np.random.normal(0, np.sqrt(2 / in_features), (out_features, in_features))
            self.biases = np.zeros(out_features)
        elif init_method == "Xavier":
            # Xavier initialization - https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            # ideal for linear activation layers; non-ideal for non-linear activation layers (i.e., ReLU)
            self.weights = np.random.uniform(-1/np.sqrt(in_features), 1/np.sqrt(in_features), (out_features, in_features))
            self.biases = np.zeros(out_features)
        elif init_method == "XavierNorm":
            # Normalized Xavier initialization - https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            self.weights = np.random.uniform(-np.sqrt(6)/np.sqrt(in_features+out_features), np.sqrt(6)/np.sqrt(in_features+out_features), (out_features, in_features))
            self.biases = np.zeros(out_features)
        else:
            raise Exception("Initialization technique not recognized.")

        # Gradient descent initialization
        self.gradWeights = np.zeros_like(self.weights)
        self.gradBiases = np.zeros_like(self.biases)

    def forward(self, _input):
        # input is NxF where F is number of in features, and N is number of data samples
        return np.dot(_input, self.weights.T) + self.biases

    def backward(self, _input, _gradPrev):
        """
        _gradPrev has shape N x out_features

        Assume dL/dY has already been computed, a.k.a. _gradPrev
        dL/dX = dL/dY * W.T
        dL/dW = _input.T * dL/dY
        dL/dB = sum(dL/dY, axis=0)
        """
        self.gradWeights += _gradPrev.T.dot(_input) / _input.shape[0]
        self.gradBiases += np.sum(_gradPrev, axis=0) / _input.shape[0]

        return np.dot(_gradPrev, self.weights)

    def params(self):
        return [self.weights, self.biases], [self.gradWeights, self.gradBiases]

    def type(self):
        return "Linear Layer"


class Dropout(Module):
    def __init__(self, p=0.5) -> None:
        """
        This class is implementing "inverse dropout" which can prevent the
        explosion or saturation of neurons. See https://bit.ly/3Ipmg12 for more info

        This is preferred to scaling during test-time
        """
        super().__init__()
        self.p = p # probability of keeping some unit active; higher p = less dropout

    def forward(self, _input):
        _output = _input
        if self.p > 0 and self.train:
            self.mask = np.random.binomial(n=1, p=self.p, size=_input.shape) / self.p
            _output *= self.mask
        return _output

    def backward(self, _input, _gradPrev):
        # scale the backwards pass by the same amount
        _output = _gradPrev
        if self.p > 0 and self.train:
            _output *= self.mask
        return _output

    def params(self):
        return None, None

    def type(self):
        return "Dropout Layer"
