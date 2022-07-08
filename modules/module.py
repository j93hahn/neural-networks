class Module():
    def __init__(self) -> None:
        return

    def forward(self, _input):
        pass

    def backward(self, _input, _gradPrev):
        pass
        """
        _gradPrev is the gradient/delta of the previous layer in the Sequential
            model when applying backwardagation

        return self._gradCurr multiplies the gradient of the current layer and
            passes it to the next layer in the sequence
        """

    def params(self):
        """
        Return the parameters and their gradients
        """
        pass

    def train(self):
        self.train = True

    def eval(self):
        self.train = False

    def type(self):
        pass
