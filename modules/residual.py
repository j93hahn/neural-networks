from .module import Module
import numpy as np


class ResidualBlock(Module):
    def __init__(self, *args: Module) -> None:
        super().__init__()
        """
        Implementing the residual block as described in He 15
        - Equation: H(x) = F(x) + x where F(x) is the residual

        Initial implementation is designed specifically for MLP (non-convolutional)
        """
        self.layers = [*args]

    def size(self):
        return len(self.layers)

    def components(self):
        for i in range(self.size()):
            print(self.layers[i].name())

    def forward(self, _input):
        self._inputs = [_input]
        for i in range(self.size()):
            self._inputs.append(self.layers[i].forward(self._inputs[i]))
        _output = self._inputs[-1] # F(x)
        _output += _input # F(x) + x
        return _output

    def backward(self, _input, _gradPrev):
        self._gradPrevArray = [0] * (self.size() + 1)
        self._gradPrevArray[self.size()] = _gradPrev
        for i in reversed(range(self.size())):
            self._gradPrevArray[i] = self.layers[i].backward(self._inputs[i], self._gradPrevArray[i + 1])
        self._adjoint = self._gradPrevArray[0]
        return self._adjoint

    def params(self):
        params = []
        gradParams = []
        for layer in self.layers:
            _p, _g = layer.params()
            if _p is not None:
                params.append(_p)
                gradParams.append(_g)
        return params, gradParams

    def name(self):
        return "Residual Block"
