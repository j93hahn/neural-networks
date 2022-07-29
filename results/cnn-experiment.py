# process training and testing data here
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])


batch_size = 100
test_size = 1
epochs = 12
count = 50 # how often we should save information to disk


trainset = FashionMNIST(root='./data', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = FashionMNIST(root='./data', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=test_size, shuffle=True, num_workers=2)


# training and inference
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from summary import summary, summary_string

# specify normalization technique here
#from torch.nn import BatchNorm2d as Norm
#from torch.nn import LayerNorm as Norm
from torch.nn import GroupNorm as Norm
from tqdm import tqdm


def LeNet():
    groups = -1
    if type(Norm) == nn.GroupNorm:
        groups = 2
    else:
        groups = 1
    model = nn.Sequential(
        nn.Conv2d(1, 6, 3, stride=1, padding=1, groups=groups),

        #Norm(6),
        #Norm([6, 28, 28]),
        #Norm(num_groups=groups, num_channels=6),

        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, 3, stride=1, padding=1, groups=groups),

        #Norm(16),
        #Norm([16, 14, 14]),
        #Norm(num_groups=groups, num_channels=16),

        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(784, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    param_dict = {k:[] for k,_ in model.named_parameters()}
    grad_dict = {k:[] for k,_ in model.named_parameters()}
    return model, criterion, optimizer, param_dict, grad_dict


def VGG():
    groups = -1
    if type(Norm) == nn.GroupNorm:
        groups = 2
    else:
        groups = 1
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, stride=1, padding=1, groups=groups),

        #Norm(8),
        #Norm([8, 28, 28]),
        Norm(num_groups=groups, num_channels=8),

        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(8, 24, 3, stride=1, padding=1, groups=groups),

        #Norm(24),
        #Norm([24, 14, 14]),
        Norm(num_groups=groups, num_channels=24),

        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(24, 72, 3, stride=1, padding=1, groups=groups),

        #Norm(72),
        #Norm([72, 7, 7]),
        Norm(num_groups=groups, num_channels=72),

        nn.ReLU(),
        nn.MaxPool2d(kernel_size=7, stride=7),
        nn.Flatten(),
        nn.Linear(72, 10)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    param_dict = {k:[] for k,_ in model.named_parameters()}
    grad_dict = {k:[] for k,_ in model.named_parameters()}
    return model, criterion, optimizer, param_dict, grad_dict


def init_params(layer):
    # https://pytorch.org/docs/stable/nn.init.html
    if type(layer) == nn.Linear:
        nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.uniform_(layer.bias)
    elif type(layer) == nn.Conv2d:
        nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.uniform_(layer.bias)
    elif type(layer) == Norm:
        nn.init.uniform_(layer.weight)
        nn.init.uniform_(layer.bias)


def process_dict(numeric_dict):
    for k in numeric_dict.keys():
        ele = torch.stack(numeric_dict[k]).detach().numpy()
        numeric_dict[k] = ele.reshape(epochs, int(len(trainloader) / count), -1)


def checkpoint(param_dict, grad_dict):
    process_dict(param_dict)
    process_dict(grad_dict)
    torch.save(param_dict, 'experiments/weightnorm/vgg-kaiunifi-gn/param.pt')
    torch.save(grad_dict, 'experiments/weightnorm/vgg-kaiunifi-gn/grad.pt')


def retrieve_numeric_values(model, mode, numeric_dict):
    for k,v in model.named_parameters():
        if mode == "params":
            numeric_dict[k].append(v.reshape(-1))
        elif mode == "gradients":
            numeric_dict[k].append(v.grad.reshape(-1))
        else:
            raise Exception("Invalid mode specified")


def io_summary(model):
    with open('experiments/weightnorm/vgg-kaiunifi-gn/summary.txt', 'w') as f:
        result, _ = summary_string(model, (1, 28, 28), device="cpu")
        f.write(result)
    f.close()
    print("Torchsummary successfully exported")


def training(model, criterion, optimizer, param_dict, grad_dict, collect=True):
    model.train()
    losses = []
    for e in range(epochs):
        print("-- Beginning Training Epoch " + str(e + 1) + " --")
        epoch_losses = []
        for step, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            # collect parameter values before optimizer.step()
            if step % count == 0 and collect:
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
            if step % count == 0 and collect:
                retrieve_numeric_values(model, "gradients", grad_dict)

        losses.append(torch.stack(epoch_losses))

    print("Training completed...")
    np.save('experiments/weightnorm/vgg-kaiunifi-gn/loss.npy', torch.stack(losses).detach().numpy())


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
    print("Inference completed, loss rate: {}".format(loss))


def main():
    print("Batch size: " + str(batch_size))
    model, criterion, optimizer, param_dict, grad_dict = VGG()
    io_summary(model)
    summary(model, (1, 28, 28), device="cpu")
    model.apply(init_params)

    training(model, criterion, optimizer, param_dict, grad_dict, True)
    checkpoint(grad_dict, param_dict)
    print("Numeric processing completed, now beginning inference...")
    inference(model)


if __name__ == '__main__':
    main()
