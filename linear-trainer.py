import numpy as np
import modules as m
from pudb import set_trace
from data_loaders import mnist
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


input_size = 50


class L2Loss():
    def __init__(self) -> None:
        pass

    def loss(self, _input, _labels):
        #return np.sqrt(np.sum(np.square(_input)))
        #set_trace()
        self._output = np.sum(np.square(_labels - (_input - _input.max())))
        return self._output

    def backward(self, _input, _labels):
        return _input.squeeze(axis=0) * 2
        #return self._output * -2


model = m.Sequential(
    m.Linear(784, 16),
    m.ReLU(),
    m.Linear(16, 10),
)

#m.Sequential(
#    m.Linear(input_size, input_size)
#)


#loss = L2Loss()
#loss = m.MSELoss()
#loss = m.CrossEntropyLoss()
loss = m.SoftMaxLoss()


def viz(ii, errors):
    plt.plot(ii, errors)
    plt.title("L2Loss for Linear Layer")
    plt.xlabel("Iteration Number")
    plt.ylabel("L2Loss")
    plt.savefig("plots/l2loss_plot.png")
    plt.show()


def main():
    model.train()
    T = 10000
    

    """model.train()
    T = 10000
    input = np.random.randint(low=0, high=2, size=(1, input_size))[:, np.newaxis]
    correct = input.copy()
    correct[correct == 0] = -1
    ii = np.arange(0, T + 1, 100)
    errors = np.zeros_like(ii, dtype=np.int64)
    init_error = loss.loss(model.forward(input), correct)
    print("Initial Error: " + str(init_error))
    #set_trace()

    for t in tqdm(range(T + 1)):
        #if t == 0 or t == T:
        #    set_trace()
        predictions = model.forward(input)
        error = loss.loss(predictions, correct)
        #if t % 100 == 0:
            #errors[t // 100] = error
        #set_trace()
        #set_trace()
        model.backward(input, loss.backward(predictions, correct))
        #model.update_params(t)

    #set_trace()
    end_error = loss.loss(model.forward(input), correct)
    print("End Error: " + str(end_error))
    viz(ii, errors)

    #slope = (errors[-1] - errors[0]) / (ii[-1] - ii[0])
    #print("Overall Slope: ", slope) # Currently generating 0.0"""


if __name__ == '__main__':
    main()