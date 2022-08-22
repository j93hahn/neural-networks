"""
Here's a simple exercise: train a standard CNN architecture but freeze the conv
layers and only train the linear classification layers. How well can the model
learn?
"""
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Train, evaluate, and store data from convolutional models')
    parser.add_argument(
        '-i',
        required=True,
        choices=['n', 'u', 'xn', 'xu', 'ku'],
        help='Initialization technique')
    parser.add_argument(
        '-fan',
        choices=['in', 'out'],
        help='Fan in or fan out, only for kaiming uniform initialization')
    parser.add_argument(
        '-e',
        required=True,
        choices=['one', 'ten'],
        help='Number of epochs to train the model under')
    parser.add_argument(
        '-freeze',
        required=True,
        choices=['conv', 'linear', 'none'],
        help='Determine which segments of the CNN to freeze the gradients of')

    args = vars(parser.parse_args())

    epochs = 0
    save_location = 'epoch_'
    if args['e'] == 'one':
        epochs = 1
        save_location += '1'
    elif args['e'] == 'ten':
        epochs = 10
        save_location += '10'

    if args['i'] == 'ku':
        save_location += '/' + args['i'] + '-' + args['fan'] + '-' + args['freeze'] + '.npy'
    else:
        save_location += '/' + args['i'] + '-' + args['freeze'] + '.npy'

    return args, epochs, save_location


args, epochs, save_location = parse_args()
batch_size = 100
test_size = 1


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
        nn.init.zeros_(layer.bias) # always initialize biases to 0
        if args['i'] == 'n':
            nn.init.normal_(layer.weight)
        elif args['i'] == 'u':
            nn.init.uniform_(layer.weight, a=-1, b=1)
        elif args['i'] == 'xn':
            nn.init.xavier_normal_(layer.weight)
        elif args['i'] == 'xu':
            nn.init.xavier_uniform_(layer.weight)
        elif args['i'] == 'ku':
            mode = 'fan_in' if args['fan'] == 'in' else 'fan_out'
            nn.init.kaiming_uniform_(layer.weight, mode=mode, nonlinearity='relu')


# freeze Conv2d layer parameters here
def freeze_conv_layers(model):
    for index, layer in model.named_children():
        if type(layer) == nn.Conv2d:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False


# freeze Linear layer parameters here
def freeze_linear_layers(model):
    for index, layer in model.named_children():
        if type(layer) == nn.Linear:
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

    with open('results.txt', 'a') as f:
        f.write("{}".format(loss))
        f.write("\n")
    f.close()
    print(loss)


if __name__ == '__main__':
    model, criterion, optimizer = build_model()
    model.apply(init_params)

    if args['freeze'] == 'conv':
        freeze_conv_layers(model)
    elif args['freeze'] == 'linear':
        freeze_linear_layers(model)

    losses = training(model, criterion, optimizer)
    np.save(save_location, losses)
    inference(model)
