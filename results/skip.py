"""
Here's a simple exercise: train a standard CNN architecture but freeze the conv
layers and only train the linear classification layers. How well can the model
learn?
"""

batch_size = 100
test_size = 1
epochs = 1
count = 50 # how often we should save information to disk


# process training and testing data here
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])


trainset = FashionMNIST(root='./data', train=True, download=False, transform=transform)
testset = FashionMNIST(root='./data', train=False, download=False, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=test_size, shuffle=True, num_workers=2)


# training and inference
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm


def build_model():
    model = nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(64, 128, 3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(128, 256, 3, padding=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(7, 7),

        nn.Flatten(),
        nn.Linear(256, 10)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    return model, criterion, optimizer


def init_params(layer):
    if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
        nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(layer.bias)


# freeze Conv2d layer parameters here
def freeze(model):
    for index, layer in model.named_children():
        if type(layer) == nn.Conv2d:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False


def training(model, criterion, optimizer):
    model.train()
    losses = []
    for e in range(epochs):
        print("-- Beginning Training Epoch " + str(e + 1) + " --")
        epoch_losses = []
        for step, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            # execute mini-batch
            inputs, _labels = data
            optimizer.zero_grad()

            labels = torch.zeros((batch_size, 10))
            labels[torch.arange(0, batch_size), _labels] = 1

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # collect loss of mini-batch
            epoch_losses.append(loss)

        losses.append(torch.stack(epoch_losses))

    print("Training completed...")
    return torch.stack(losses).detach().numpy()


def inference(model):
    model.eval()
    accuracy = 0
    total = 10000
    with torch.no_grad():
        for _, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
            inputs, _labels = data

            labels = torch.zeros((test_size, 10))
            labels[torch.arange(0, test_size), _labels] = 1

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _, correct = torch.max(labels.data, 1)
            accuracy += 1 if correct == predicted else 0
    loss = float("{0:.4f}".format(1 - accuracy/total))
    print(loss)


if __name__ == '__main__':
    model, criterion, optimizer = build_model()
    model.apply(init_params)
    freeze(model)

    losses = training(model, criterion, optimizer)
    np.save('losses1.npy', losses) # loss1: 0.2619 (!)
    inference(model)
