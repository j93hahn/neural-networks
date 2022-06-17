import sys
sys.path.insert(0, '../mlp-arch/')

import numpy as np
from pudb import set_trace
from model import Base, Sequential, Linear, ReLU


class Conv2d(Base):
    def __init__(self) -> None:
        pass

    def loss():
        pass


# deep residual network class (ResNet)
class ResNet():
    def __init__(self) -> None:
        pass


model = Sequential()
model.add(Linear(784, 32))
model.add(ReLU())
model.add(Linear(32, 16))
#print(model.components())
model.components()