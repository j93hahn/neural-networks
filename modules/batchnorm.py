from module import Module
import numpy as np


# requires mini-batch gradient descent
class BatchNorm1d(Module):
    def __init__(self, input_dim, eps=1e-5, momentum=0.1) -> None:
        super().__init__()
        """
        Input has dimension m by C, where m is the batchsize

        To calculate running statistics, use the following formulas:
        E[x] <-- np.mean(mean_beta)
        Var[x] <-- m/(m-1)*np.mean(var_beta)
        """
        self.train_first = True
        self.inf_first = True
        self.eps = eps
        self.momentum = momentum
        self.count = 0
        self.gamma = np.ones(input_dim)
        self.beta = np.zeros(input_dim)
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def forward(self, _input):
        if self.train: # training batch-normalized network
            if self.train_first: # retrieve shape of input from the first mini-batch
                self.m = _input.shape[0] # batch-size
                self.running_mean = np.zeros((self.m, 1))
                self.running_var = np.zeros((self.m, 1))
                self.train_first = False

            # calculate mini-batch statistics here
            mean = np.mean(_input, axis=1)[:, np.newaxis] # m by 1
            var = np.mean(np.square(_input - mean), axis=1)[:, np.newaxis] # m by 1

            # normalize data, then scale and shift via affine parameters
            x_hat = (_input - mean) / np.sqrt(var + self.eps) # m by C, broadcasted
            y = self.gamma * x_hat + self.beta # m by C, broadcasted

            # update moving statistics
            self.running_mean += mean
            self.running_var += var

            # return output values
            self.count += 1
            return y

        else: # inference stage
            if self.inf_first:
                self.running_mean /= self.count
                self.running_var /= self.count
                self.running_var *= (self.m / (self.m - 1))
                self.count = 0 # training is over
                self.inf_first = False

            y = self.gamma * _input/(np.sqrt(self.running_var + self.eps))
            y += self.beta - ((self.gamma * self.running_mean) / (np.sqrt(self.running_var + self.eps)))
            return y

    def backward(self, _gradPrev):
        """

        """
        ...

    def params(self):
        return [self.gamma, self.beta], [self.gradGamma, self.gradBeta]

    def type(self):
        return "Batch Normalization Layer"



test = BatchNorm1d()
breakpoint()
test.forward(np.random.randn(40, 100))

test.eval()
breakpoint()
test.forward(np.random.randn(40, 100))
