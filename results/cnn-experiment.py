# process training and testing data here
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchsummary import summary


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
    param_dict = {k:[] for k,_ in model.named_parameters()}
    grad_dict = {k:[] for k,_ in model.named_parameters()}
    return model, criterion, optimizer, param_dict, grad_dict


def init_params(layer):
    # https://pytorch.org/docs/stable/nn.init.html
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d or type(layer) == Norm:
        nn.init.uniform_(layer.weight)
        nn.init.uniform_(layer.bias)


def process_dict(numeric_dict):
    for k in numeric_dict.keys():
        ele = torch.stack(numeric_dict[k]).detach().numpy()
        numeric_dict[k] = ele.reshape(epochs, int(len(trainloader) / count), -1)


def checkpoint(param_dict, grad_dict):
    process_dict(param_dict)
    process_dict(grad_dict)
    torch.save(param_dict, 'experiments/weightinit/uniform/param.pt')
    torch.save(grad_dict, 'experiments/weightinit/uniform/grad.pt')


def retrieve_numeric_values(model, mode, numeric_dict):
    for k,v in model.named_parameters():
        if mode == "params":
            numeric_dict[k].append(v.reshape(-1))
        elif mode == "gradients":
            numeric_dict[k].append(v.grad.reshape(-1))
        else:
            raise Exception("Invalid mode specified")


def training(model, criterion, optimizer, param_dict, grad_dict):
    model.train()
    losses = []
    for e in range(epochs):
        print("-- Beginning Training Epoch " + str(e + 1) + " --")
        epoch_losses = []
        for step, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            # collect parameter values before optimizer.step()
            if step % count == 0:
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
            if step % count == 0:
                retrieve_numeric_values(model, "gradients", grad_dict)

        losses.append(torch.stack(epoch_losses))

    print("Training completed, now processing numeric values for visualizations...")
    np.save('experiments/weightinit/uniform/loss.npy', torch.stack(losses).detach().numpy())


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
    model, criterion, optimizer, param_dict, grad_dict = build_model()
    summary(model, (1, 28, 28), device="cpu")
    model.apply(init_params)

    training(model, criterion, optimizer, param_dict, grad_dict)
    checkpoint(grad_dict, param_dict)
    print("Numeric processing completed, now beginning inference...")
    inference(model)


if __name__ == '__main__':
    main()
