from .module import Module


class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        super().__init__()
        self.layers = [*args]

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

    def backward(self, _input, _gradPrev):
        self._gradPrevArray = [0] * (self.size() + 1)
        self._gradPrevArray[self.size()] = _gradPrev
        for i in reversed(range(self.size())):
            # think about the adjoint state
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

    def train(self):
        self.train
        for layer in self.layers:
            layer.train

    def eval(self):
        self.eval
        for layer in self.layers:
            layer.eval()
