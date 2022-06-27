from .optimizer import Optimizer
import numpy as np


class Adam(Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8) -> None:
        super().__init__()
