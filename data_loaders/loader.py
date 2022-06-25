# sourced and modified from: https://github.com/hsjeong5/MNIST-for-Numpy
import pickle
import numpy as np


class MNISTDataLoader():
    def mnist(self): # MNIST dataset
        with open("data_loaders/mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        self.train_images = mnist["training_images"]
        self.train_labels = mnist["training_labels"]
        self.test_images = mnist["test_images"]
        self.test_labels = mnist["test_labels"]
        self.name = "MNIST"
        return self

    def fashion_mnist(self): # fashion MNIST dataset
        with open("data_loaders/fashion_mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        self.train_images = mnist["training_images"]
        self.train_labels = mnist["training_labels"]
        self.test_images = mnist["test_images"]
        self.test_labels = mnist["test_labels"]
        self.name = "Fashion MNIST"
        return self


class CifarDataLoader():
    def __init__(self, file) -> None:
        self.batch = pickle.load(open("data_loaders/" + file, 'rb'), encoding='bytes')
        self.name = self.batch[b'batch_label']
        self.labels = np.array(self.batch[b'labels'])
        self.data = self.batch[b'data']
        self.files = self.batch[b'filenames']


mnist = MNISTDataLoader().mnist()
fashion_mnist = MNISTDataLoader().fashion_mnist()

batch = CifarDataLoader("train5")
data, labels, name, files = batch.data, batch.labels, batch.name, batch.files
