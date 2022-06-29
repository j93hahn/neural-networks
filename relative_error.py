import numpy as np
import modules as m
from data_loaders import mnist


"""
This file is used to test the relative errors of gradient descent
"""


def relative_error(x, y, h):
    h = h or 1e-12
    if type(x) is np.ndarray and type(y) is np.ndarray:
        top = np.abs(x - y)
        bottom = np.maximum(np.abs(x) + np.abs(y), h)
        return np.amax(top/bottom)
    else:
        return abs(x - y) / max(abs(x) + abs(y), h)


def numeric_gradient(f, x, df, eps):
    df = df or 1.0
    eps = eps or 1e-8
    n = x.size
    x_flat = x.reshape(n)
    dx_num = np.zeros(x.shape)
    dx_num_flat = dx_num.reshape(n)
    for i in range(n):
        orig = x_flat[i]

        x_flat[i] = orig + eps
        pos = f(x)
        if type(df) is np.ndarray:
            pos = pos.copy()

        x_flat[i] = orig - eps
        neg = f(x)
        if type(df) is np.ndarray:
            neg = neg.copy()

        d = (pos - neg) * df / (2 * eps)

        dx_num_flat[i] = d
        x_flat[i] = orig
    return dx_num


class TestCriterion(object):
    def __init__(self):
        return

    def forward(self, _input, _target):
        return np.mean(np.sum(np.abs(_input), 1))

    def backward(self, _input, _target):
        self._gradInput = np.sign(_input) / len(_input)
        return self._gradInput


# relative errors should be on the order of 1e-6 or smaller
def test_relative_error(model):

    model.eval()

    crit = TestCriterion()
    gt = np.random.random((3,10))
    x = np.random.random((3,10))
    def test_f(x):
        return crit.forward(model.forward(x), gt)

    gradInput = model.backward(x, crit.backward(model.forward(x), gt))
    gradInput_num = numeric_gradient(test_f, x, 1, 1e-6)
    print(relative_error(gradInput, gradInput_num, 1e-8))


def main():
    model = m.Linear(10, 10)
    print("testing Linear layer")
    test_relative_error(model)


    model = m.ReLU()
    print("testing ReLU layer")
    test_relative_error(model)


    model = m.Sequential(
        m.Linear(10, 10),
        m.ReLU(),
        m.Linear(10, 10)
    )
    print("testing two layer model")
    test_relative_error(model)


if __name__ == '__main__':
    main()
