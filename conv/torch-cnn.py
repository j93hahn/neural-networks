import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d()