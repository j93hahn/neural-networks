# process training and testing data here
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])


batch_size = 100
test_size = 1


trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=False, transform=transform)
testloader = DataLoader(testset, batch_size=test_size, shuffle=True, num_workers=2)


# training and inference
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr

from tqdm import tqdm


def build_model():
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3),
        nn.BatchNorm2d(num_features=4),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(4, 8, 3),
        nn.BatchNorm2d(num_features=8),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(200, 10)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = lr.StepLR(optimizer, step_size=5, gamma=0.1)
    return model, criterion, optimizer, scheduler


def training(model, criterion, optimizer, scheduler):
    model.train()
    epochs = 15
    for e in range(epochs):
        print("-- Beginning Training Epoch " + str(e + 1) + " --")
        for _, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            inputs, _labels = data
            optimizer.zero_grad()

            # inputs = inputs.reshape(batch_size, 784)
            labels = torch.zeros((batch_size, 10))
            labels[torch.arange(0, batch_size), _labels] = 1

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()


def inference(model):
    model.eval()
    accuracy = 0
    total = 0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
            inputs, _labels = data

            # inputs = inputs.reshape(test_size, 784)
            labels = torch.zeros((test_size, 10))
            labels[torch.arange(0, test_size), _labels] = 1

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            _, correct = torch.max(labels.data, 1)
            total += test_size
            accuracy += 1 if correct == predicted else 0
    loss = float("{0:.4f}".format(1 - accuracy/total))
    print("Loss rate: {}".format(loss))


def main():
    model, criterion, optimizer, scheduler = build_model()
    training(model, criterion, optimizer, scheduler)
    print("Training completed, now beginning inference...")
    inference(model)


if __name__ == '__main__':
    main()
