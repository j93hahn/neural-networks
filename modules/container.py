from .module import Module


class Sequential(Module):
    def __init__(self, *args: Module) -> None:
        super().__init__()
        self.layers = []
        for arg in args:
            self.add(arg)

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
        self.train
        for layer in self.layers:
            layer.train

    def eval(self):
        self.eval
        for layer in self.layers:
            layer.eval()
