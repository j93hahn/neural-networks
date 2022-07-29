# parse arguments first
import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train, evaluate, and store data from convolutional models')
    parser.add_argument(
        '-m',
        required=True,
        choices=['lenet', 'vgg'],
        help='Model category')
    parser.add_argument(
        '-c',
        required=True,
        choices=['i', 'n'],
        help='Category this experiment is classified under')
    parser.add_argument(
        '-i',
        required=True,
        choices=['z', 'o', 'n', 'u', 'xn', 'xu', 'ku'],
        help='Initialization technique')
    parser.add_argument(
        '-n',
        required=True,
        choices=['nn', 'bn', 'ln', 'gn'],
        help='Normalization technique')
    parser.add_argument(
        '-f',
        choices=['in', 'out'],
        help='Fan in or fan out, only for kaiming uniform initialization')
    parser.add_argument(
        '--numeric',
        action='store_true',
        help='Store parameters, gradients, and loss to file')
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Write summary and accuracy loss to file')
    parser.add_argument(
        '--print',
        action='store_true',
        help='Print model specifications to terminal')

    args = vars(parser.parse_args())
    category = 'weightinit' if args['c'] == 'i' else 'weightnorm'
    base_location = -1

    if args['i'] != 'ku' and args['f'] != None:
        raise Exception(
            "Fan in or fan out only allowed with Kaiming Uniform initialization")
    if args['i'] == 'ku' and args['f'] == None:
        raise Exception(
            "Kaiming Uniform initalization requires specification of fan in or fan out")
    if args['numeric'] or args['summary']:
        method = ''
        if args['i'] == 'z':
            method = 'zeros'
        elif args['i'] == 'o':
            method = 'ones'
        elif args['i'] == 'n':
            method = 'normal'
        elif args['i'] == 'u':
            method = 'uniform'
        elif args['i'] == 'xn':
            method = 'xaviernorm'
        elif args['i'] == 'xu':
            method = 'xavieruniform'
        elif args['i'] == 'ku':
            method = 'kaiminguniform'

        import os
        base_location = 'experiments/' + category + '/' + args['m'] + '-' + \
            method + '-' + args['n'] + '/'
        try:
            os.mkdir(base_location)
        except FileNotFoundError:
            os.makedirs(base_location)
    if args['print']:
        print("Category: " + category)
        print("Model: " + args['m'])
        print("Init: " + args['i'])
        print("Norm: " + args['n'])

    return args, base_location


# process training and testing data here
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])


args, base_location = parse_args()
batch_size = 100
test_size = 1
epochs = 12
count = 50 # how often we should save information to disk
groups = 1 if args['n'] != 'gn' else 2


trainset = FashionMNIST(root='./data', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = FashionMNIST(root='./data', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=test_size, shuffle=True, num_workers=2)


# training and inference
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from summary import summary_string

# specify normalization technique here
from tqdm import tqdm


def build_model():
    if args['m'] == 'lenet':
        layers = [nn.Conv2d(1, 6, 3, stride=1, padding=1)]

        if args['n'] == 'bn':
            layers.append(nn.BatchNorm2d(6))
        elif args['n'] == 'ln':
            layers.append(nn.LayerNorm([6, 28, 28]))
        elif args['n'] == 'gn':
            layers.append(nn.GroupNorm(num_groups=groups, num_channels=6))

        layers.extend((
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 3, stride=1, padding=1, groups=groups)
        ))

        if args['n'] == 'bn':
            layers.append(nn.BatchNorm2d(16))
        elif args['n'] == 'ln':
            layers.append(nn.LayerNorm([16, 14, 14]))
        elif args['n'] == 'gn':
            layers.append(nn.GroupNorm(num_groups=groups, num_channels=16))

        layers.extend((
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(784, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        ))

        model = nn.Sequential(*layers)
    elif args['m'] == 'vgg':
        layers = [nn.Conv2d(1, 8, 3, stride=1, padding=1)]

        if args['n'] == 'bn':
            layers.append(nn.BatchNorm2d(8))
        elif args['n'] == 'ln':
            layers.append(nn.LayerNorm([8, 28, 28]))
        elif args['n'] == 'gn':
            layers.append(nn.GroupNorm(num_groups=groups, num_channels=28))

        layers.extend((
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 24, 3, stride=1, padding=1, groups=groups)
        ))

        if args['n'] == 'bn':
            layers.append(nn.BatchNorm2d(24))
        elif args['n'] == 'ln':
            layers.append(nn.LayerNorm([24, 14, 14]))
        elif args['n'] == 'gn':
            layers.append(nn.GroupNorm(num_groups=groups, num_channels=24))

        layers.extend((
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(24, 72, 3, stride=1, padding=1, groups=groups)
        ))

        if args['n'] == 'bn':
            layers.append(nn.BatchNorm2d(72))
        elif args['n'] == 'ln':
            layers.append(nn.LayerNorm([72, 7, 7]))
        elif args['n'] == 'gn':
            layers.append(nn.GroupNorm(num_groups=groups, num_channels=72))

        layers.extend((
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=7, stride=7),
            nn.Flatten(),
            nn.Linear(72, 10)
        ))

        model = nn.Sequential(*layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    param_dict = {k:[] for k,_ in model.named_parameters()}
    grad_dict = {k:[] for k,_ in model.named_parameters()}
    return model, criterion, optimizer, param_dict, grad_dict


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
        nn.init.uniform_(layer.weight)
        nn.init.uniform_(layer.bias)
    elif args['i'] == 'xn':
        if type(layer) in conv_layers:
            nn.init.xavier_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.normal_(layer.bias)
        elif type(layer) in norm_layers:
            nn.init.normal_(layer.weight)
            nn.init.normal_(layer.bias)
    elif args['i'] == 'xu':
        if type(layer) in conv_layers:
            nn.init.xavier_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.uniform_(layer.bias)
        elif type(layer) in norm_layers:
            nn.init.uniform_(layer.weight)
            nn.init.uniform_(layer.bias)
    elif args['i'] == 'ku':
        mode = 'fan_in' if args['f'] == 'in' else 'fan_out'
        if type(layer) in conv_layers:
            nn.init.kaiming_uniform_(layer.weight, mode=mode, nonlinearity='relu')
            nn.init.uniform_(layer.bias)
        elif type(layer) in norm_layers:
            nn.init.uniform_(layer.weight)
            nn.init.uniform_(layer.bias)


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
    if args['summary']:
        with open(base_location + 'summary.txt', 'a') as f:
            f.write("Inference completed, loss rate: {}".format(loss))
        f.close()
        print("Loss successfully exported to file")
    if args['print']:
        print("Inference completed, loss rate: {}".format(loss))


def main():
    model, criterion, optimizer, param_dict, grad_dict = build_model()
    io_summary(model)
    model.apply(init_params)

    training(model, criterion, optimizer, param_dict, grad_dict)
    checkpoint(grad_dict, param_dict)
    inference(model)


if __name__ == '__main__':
    main()
