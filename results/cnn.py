# process training and testing data here
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchsummary import summary


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])


batch_size = 100
test_size = 1
epochs = 1


trainset = FashionMNIST(root='./data', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = FashionMNIST(root='./data', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=test_size, shuffle=True, num_workers=2)


# training and inference
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# specify normalization technique here
from torch.nn import BatchNorm2d as Norm
from tqdm import tqdm


def build_model():
    model = nn.Sequential(
        nn.Conv2d(1, 6, 3, stride=1, padding=1),
        Norm(6),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, 3, stride=1, padding=1),
        Norm(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(784, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return model, criterion, optimizer


def init_params(layer):
    # https://pytorch.org/docs/stable/nn.init.html
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d or type(layer) == Norm:
        nn.init.uniform_(layer.weight)
        nn.init.uniform_(layer.bias)


# store parameters from model
class Parameters():
    def __init__(self) -> None:
        self._lw9 = []; self._lb9 = []; self._lw11 = []; self._lb11 = []
        self._cw0 = []; self._cb0 = []; self._cw4 = []; self._cb4 = []
        self._nw1 = []; self._nb1 = []; self._nw5 = []; self._nb5 = []


# store gradients from model
class Gradients():
    def __init__(self) -> None:
        self._lgw9 = []; self._lgb9 = []; self._lgw11 = []; self._lgb11 = []
        self._cgw0 = []; self._cgb0 = []; self._cgw4 = []; self._cgb4 = []
        self._ngw1 = []; self._ngb1 = []; self._ngw5 = []; self._ngb5 = []


allParams = Parameters()
allGradients = Gradients()


def retrieve_numeric_values(model, mode):
    if mode == "params":
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                if name == '9':
                    allParams._lw9.append(layer.weight.reshape(-1).detach().numpy())
                    allParams._lb9.append(layer.bias.reshape(-1).detach().numpy())
                elif name == '11':
                    allParams._lw11.append(layer.weight.reshape(-1).detach().numpy())
                    allParams._lb11.append(layer.bias.reshape(-1).detach().numpy())
                else:
                    raise Exception("Invalid name type")
            elif isinstance(layer, nn.Conv2d):
                if name == '0':
                    allParams._cw0.append(layer.weight.reshape(-1).detach().numpy())
                    allParams._cb0.append(layer.bias.reshape(-1).detach().numpy())
                elif name == '4':
                    allParams._cw4.append(layer.weight.reshape(-1).detach().numpy())
                    allParams._cb4.append(layer.bias.reshape(-1).detach().numpy())
                else:
                    raise Exception("Invalid name type")
            elif isinstance(layer, Norm):
                if name == '1':
                    allParams._nw1.append(layer.weight.reshape(-1).detach().numpy())
                    allParams._nb1.append(layer.bias.reshape(-1).detach().numpy())
                elif name == '5':
                    allParams._nw5.append(layer.weight.reshape(-1).detach().numpy())
                    allParams._nb5.append(layer.bias.reshape(-1).detach().numpy())
                else:
                    raise Exception("Invalid name type")
    elif mode == "gradients":
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Linear):
                if name == '9':
                    allGradients._lgw9.append(layer.weight.grad.reshape(-1).detach().numpy())
                    allGradients._lgb9.append(layer.bias.grad.reshape(-1).detach().numpy())
                elif name == '11':
                    allGradients._lgw11.append(layer.weight.grad.reshape(-1).detach().numpy())
                    allGradients._lgb11.append(layer.bias.grad.reshape(-1).detach().numpy())
                else:
                    raise Exception("Invalid name type")
            elif isinstance(layer, nn.Conv2d):
                if name == '0':
                    allGradients._cgw0.append(layer.weight.grad.reshape(-1).detach().numpy())
                    allGradients._cgb0.append(layer.bias.grad.reshape(-1).detach().numpy())
                elif name == '4':
                    allGradients._cgw4.append(layer.weight.grad.reshape(-1).detach().numpy())
                    allGradients._cgb4.append(layer.bias.grad.reshape(-1).detach().numpy())
                else:
                    raise Exception("Invalid name type")
            elif isinstance(layer, Norm):
                if name == '1':
                    allGradients._ngw1.append(layer.weight.grad.reshape(-1).detach().numpy())
                    allGradients._ngb1.append(layer.bias.grad.reshape(-1).detach().numpy())
                elif name == '5':
                    allGradients._ngw5.append(layer.weight.grad.reshape(-1).detach().numpy())
                    allGradients._ngb5.append(layer.bias.grad.reshape(-1).detach().numpy())
                else:
                    raise Exception("Invalid name type")
    else:
        raise Exception("Invalid mode inserted")


def process_numeric_values(losses):
    # process loss values
    losses = torch.stack(losses).detach().numpy()

    # process parameters
    for attr in filter(lambda a: not a.startswith('__'), dir(allParams)):
        setattr(allParams, attr, np.array(getattr(allParams, attr)).reshape(epochs, len(trainloader), -1))

    # process gradients
    for attr in filter(lambda a: not a.startswith('__'), dir(allGradients)):
        setattr(allGradients, attr, np.array(getattr(allGradients, attr)).reshape(epochs, len(trainloader), -1))

    np.savez('experiments/weightinit/uniform.npz', losses,
        allParams._lw9, allParams._lb9, allParams._lw11, allParams._lb11,
        allParams._cw0, allParams._cb0, allParams._cw4, allParams._cb4,
        allParams._nw1, allParams._nb1, allParams._nw5, allParams._nb5,
        allGradients._lgw9, allGradients._lgb9, allGradients._lgw11, allGradients._lgb11,
        allGradients._cgw0, allGradients._cgb0, allGradients._cgw4, allGradients._cgb4,
        allGradients._ngw1, allGradients._ngb1, allGradients._ngw5, allGradients._ngb5,
    )


def training(model, criterion, optimizer):
    model.train()
    losses = []
    for e in range(epochs):
        print("-- Beginning Training Epoch " + str(e + 1) + " --")
        epoch_losses = []
        for _, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            # collect parameter values before optimizer.step()
            retrieve_numeric_values(model, "params")

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
            retrieve_numeric_values(model, "gradients")

        losses.append(torch.stack(epoch_losses))

    print("Training completed, now processing numeric values for visualizations...")
    process_numeric_values(losses)
    print("Numeric processing completed, now beginning inference...")


def inference(model):
    model.eval()
    accuracy = 0
    total = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
            inputs, _labels = data

            labels = torch.zeros((test_size, 10))
            labels[torch.arange(0, test_size), _labels] = 1

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _, correct = torch.max(labels.data, 1)
            total += test_size
            accuracy += 1 if correct == predicted else 0
    loss = float("{0:.4f}".format(1 - accuracy/total))
    print("Inference completed, loss rate: {}".format(loss))


def main():
    model, criterion, optimizer = build_model()
    summary(model, (1, 28, 28), device="cpu")
    model.apply(init_params)

    training(model, criterion, optimizer)
    inference(model)


if __name__ == '__main__':
    main()
