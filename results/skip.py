"""
Here's a simple exercise: train a standard CNN architecture but freeze the conv
layers and only train the linear classification layers. How well can the model
learn?
"""

batch_size = 100
test_size = 1
epochs = 12
count = 50 # how often we should save information to disk


# process training and testing data here
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST


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
from summary import summary, summary_string

from tqdm import tqdm


def build_model():
    breakpoint()
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

        nn.Linear(1568, 10)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    return model, criterion, optimizer


def init_params(layer):
    norm_layers = [nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm]
    conv_layers = [nn.Linear, nn.Conv2d]
    all_layers = [nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm]

    if args['i'] == 'z' and type(layer) in all_layers:
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
    elif args['i'] == 'o' and type(layer) in all_layers:
        nn.init.ones_(layer.weight)
        nn.init.ones_(layer.bias)
    elif args['i'] == 'n' and type(layer) in all_layers:
        nn.init.normal_(layer.weight)
        nn.init.normal_(layer.bias)
    elif args['i'] == 'u' and type(layer) in all_layers:
        nn.init.uniform_(layer.weight, a=-1, b=1)
        nn.init.uniform_(layer.bias, a=-1, b=1)
    elif args['i'] == 'xn':
        if type(layer) in conv_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.normal_(layer.bias)
        elif type(layer) in norm_layers:
            nn.init.normal_(layer.weight)
            nn.init.normal_(layer.bias)
    elif args['i'] == 'xu':
        if type(layer) in conv_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.uniform_(layer.bias, a=-1, b=1)
        elif type(layer) in norm_layers:
            nn.init.uniform_(layer.weight, a=-1, b=1)
            nn.init.uniform_(layer.bias, a=-1, b=1)
    elif args['i'] == 'ku':
        mode = 'fan_in' if args['f'] == 'in' else 'fan_out'
        if type(layer) in conv_layers:
            nn.init.kaiming_uniform_(layer.weight, mode=mode, nonlinearity='relu')
            nn.init.uniform_(layer.bias, a=-1, b=1)
        elif type(layer) in norm_layers:
            nn.init.uniform_(layer.weight, a=-1, b=1)
            nn.init.uniform_(layer.bias, a=-1, b=1)


def process_dict(numeric_dict):
    for k in numeric_dict.keys():
        ele = torch.stack(numeric_dict[k]).detach().numpy()
        numeric_dict[k] = ele.reshape(epochs, int(len(trainloader) / count), -1)


def checkpoint(param_dict, grad_dict):
    if args['numeric']:
        process_dict(param_dict)
        process_dict(grad_dict)
        torch.save(param_dict, base_location + 'param.pt')
        torch.save(grad_dict, base_location + 'grad.pt')
        print("Numeric processing completed, now beginning inference...")


def retrieve_numeric_values(model, mode, numeric_dict):
    for k,v in model.named_parameters():
        if mode == "params":
            x = v.clone().detach()
            numeric_dict[k].append(x.reshape(-1))
        elif mode == "gradients":
            x = v.grad.clone().detach()
            numeric_dict[k].append(x.reshape(-1))
        else:
            raise Exception("Invalid mode specified")


def io_summary(model):
    if args['summary']:
        with open(base_location + 'summary.txt', 'w') as f:
            result, _ = summary_string(model, (1, 28, 28), device="cpu")
            f.write(result)
        f.close()
        print("Torchsummary successfully exported")


def training(model, criterion, optimizer, param_dict, grad_dict):
    model.train()
    losses = []
    for e in range(epochs):
        print("-- Beginning Training Epoch " + str(e + 1) + " --")
        epoch_losses = []
        for step, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            # collect parameter values before optimizer.step()
            if step % count == 0 and args['numeric']:
                retrieve_numeric_values(model, "params", param_dict)

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

            # collect gradient values after computing the loss
            if step % count == 0 and args['numeric']:
                retrieve_numeric_values(model, "gradients", grad_dict)

        losses.append(torch.stack(epoch_losses))

    print("Training completed...")
    if args['numeric']:
        np.save(base_location + 'loss.npy', torch.stack(losses).detach().numpy())


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
    io_summary(model)
    model.apply(init_params)

    training(model, criterion, optimizer, param_dict, grad_dict)
    inference(model)