import numpy as np
import modules as m
from data_loaders import mnist


"""
This file is used to test gradient descent capabilities
"""


def sgd(x, dx, lr, weight_decay = 0):
    # standard sgd
    if type(x) is list:
        assert len(x) == len(dx), 'Should be the same'
        for _x, _dx in zip(x, dx):
            sgd(_x, _dx, lr)
    else:
        x -= lr * (dx + 2 * weight_decay * x)


def sgdm(x, dx, lr, alpha = 0.8 , state = None, weight_decay = 0):
    # sgd with momentum, standard update
    if not state:
        if type(x) is list:
            state = [None] * len(x)
        else:
            state = {}
            state['v'] = np.zeros(x.shape)
    if type(x) is list:
        for _x, _dx, _state in zip(x, dx, state):
            sgdm(_x, _dx, lr, alpha, _state)
    else:
        state['v'] *= alpha
        state['v'] += lr * (dx + 2 * weight_decay * x)
        x -= state['v']


def sgdmom(x, dx, lr, alpha = 0, state = None, weight_decay = 0):
    # sgd momentum, uses nesterov update (reference: http://cs231n.github.io/neural-networks-3/#sgd)
    if not state:
        if type(x) is list:
            state = [None] * len(x)
        else:
            state = {}
            state['m'] = np.zeros(x.shape)
            state['tmp'] = np.zeros(x.shape)
    if type(x) is list:
        for _x, _dx, _state in zip(x, dx, state):
            sgdmom(_x, _dx, lr, alpha, _state)
    else:
        state['tmp'] = state['m'].copy()
        state['m'] *= alpha
        state['m'] -= lr * (dx + 2 * weight_decay * x)

        x -= alpha * state['tmp']
        x += (1 + alpha) * state['m']


class TestCriterion(object):
    def __init__(self):
        return

    def forward(self, _input, _target):
        return np.mean(np.sum(np.abs(_input), 1))

    def backward(self, _input, _target):
        self._gradInput = np.sign(_input) / len(_input)
        return self._gradInput


# the loss should decrease continuously over time
def test_grad_descent(model, crit, trainX, test_learning_rate):
    #params, gradParams = model.params()
    it = 0
    state = None
    while True:
        params, gradParams = model.params()
        output = model.forward(trainX)
        loss = crit.forward(output, None)
        if it % 1000 == 0:
            print(loss)
        doutput = crit.backward(output, None)
        model.backward(trainX, doutput)
        #sgdm(params, gradParams, test_learning_rate, 0.8, state)
        sgdmom(params, gradParams, lr=test_learning_rate, alpha=0, state=state)
        if it > 10000:
            break
        it += 1


def main():
    trainX = np.random.random((10, 5))
    crit = TestCriterion()
    test_learning_rate = 1e-4
    model = m.Sequential(
        m.Linear(5, 3),
        m.ReLU(),
        m.Dropout(),
        m.Linear(3, 1)
    )

    print("testing gradient descent of two layer model")
    test_grad_descent(model, crit, trainX, test_learning_rate)


if __name__ == '__main__':
    main()
