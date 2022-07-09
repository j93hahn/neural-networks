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

        # initialize parameters
        self.gamma = np.ones(input_dim) # PyTorch implementation
        self.beta = np.zeros(input_dim)
        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def forward(self, _input):
        if self.train: # training batch-normalized network
            # self._input = _input
            if self.train_first: # retrieve shape of input from the first mini-batch
                self.m = _input.shape[0] # batch-size
                self.running_mean = np.zeros(_input.shape[1])
                self.running_var = np.zeros(_input.shape[1])
                self.train_first = False

            # calculate mini-batch statistics here
            self.mean = np.mean(_input, axis=0)
            self.var = np.mean(np.square(_input - self.mean), axis=0)

            # normalize data, then scale and shift via affine parameters
            self.x_hat = (_input - self.mean) / np.sqrt(self.var + self.eps) # m by C, broadcasted
            y = self.gamma * self.x_hat + self.beta # m by C, broadcasted

            # update moving statistics
            self.running_mean += self.mean
            self.running_var += self.var

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

    def backward(self, _input, _gradPrev):
        """
        All gradient calculations taken directly from Ioffe and Szegedy 2015

        Requires mean, var, x_hat to be passed from the forward pass
        """
        _gradxhat = _gradPrev * self.gamma
        _gradVar = np.sum(_gradxhat * (_input - self.mean) * -1/2*np.power(self.var + self.eps, -3/2), axis=0)
        _gradMean = np.sum(_gradxhat * -1/np.sqrt(self.var + self.eps), axis=0)
        _gradCurr = _gradxhat * 1/np.sqrt(self.var + self.eps) + _gradVar*2*(_input - self.mean)/self.m + _gradMean*1/self.m
        self.gradGamma += np.sum(_gradPrev * self.x_hat, axis=0)
        self.gradBeta += np.sum(_gradPrev, axis=0)
        return _gradCurr

    def params(self):
        return [self.gamma, self.beta], [self.gradGamma, self.gradBeta]

    def type(self):
        return "Batch Normalization Layer"


class GroupNorm1d(BatchNorm1d):
    def __init__(self, input_dim) -> None:
        super(GroupNorm1d, self).__init__(input_dim=input_dim)
        """
        Implemented exactly according to Wu and He 2018
        """

    def forward(self, _input):
        ...

    def backward(self, _input, _gradPrev):
        ...

    def type(self):
        return "Group Normalization Layer"


def test(run_batchnorm=False, run_groupnorm=True):
    if run_batchnorm:
        test = BatchNorm1d(100)
        test.train()
        breakpoint()
        test.forward(np.random.randn(40, 100))

        test.eval()
        breakpoint()
        test.forward(np.random.randn(40, 100))

        breakpoint()
        test.backward(np.random.randn(40, 100))

    if run_groupnorm:
        test = GroupNorm1d(input_dim=100)
        test.train()
        breakpoint()
        print("100")

if __name__ == '__main__':
    test(False, True)