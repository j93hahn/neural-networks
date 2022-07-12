from .module import Module
import numpy as np


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups,
                 stride=1, padding=0, padding_mode="zeros") -> None:
        super().__init__()
        """
        Input and output both have shape (N, C, H, W)
            N = batch size, C = channels
            H, W = image size

        kernel_size can be a single int (H & W) or a tuple of ints (H, W)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels # number of features that we are producing
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups # discuss with Haochen to identify relationship between
        self.padding = padding
        self.padding_mode = padding_mode
        self.weights = ... # (out_channels, in_channels/groups, kernel_size, kernel_size)
        self.biases = ... # (out_channels)
        self.gradWeights = np.zeros_like(self.weights)
        self.gradBiases = np.zeros_like(self.biases)

    def forward(self, _input):
        ...

    def backward(self, _input, _gradPrev):
        ...

    def params(self):
        return [self.weights, self.biases], [self.gradWeights, self.gradBiases]

    def name(self):
        return "Conv2d Layer"


"""
Implementing 3 standard types of pooling layers
"""
class MaxPool(Module):
    def __init__(self, stride=2) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, _input):
        ...

    def backward(self, _input, _gradPrev):
        ...

    def params(self):
        pass

    def name(self):
        return "Max Pooling Layer"


class MinPool(Module):
    def __init__(self, stride=2) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, _input):
        ...

    def backward(self, _input, _gradPrev):
        ...

    def params(self):
        pass

    def name(self):
        return "Min Pooling Layer"


class AvgPool(Module):
    def __init__(self, stride=2) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, _input):
        ...

    def backward(self, _input, _gradPrev):
        ...

    def params(self):
        pass

    def name(self):
        return "Average Pooling Layer"
