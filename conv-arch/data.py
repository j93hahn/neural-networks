# sourced and modified from: https://github.com/hsjeong5/MNIST-for-Numpy
import pickle

class DataLoader():
    def __init__(self) -> None:
        with open("mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        self.train_images = mnist["training_images"]
        self.train_labels = mnist["training_labels"]
        self.test_images = mnist["test_images"]
        self.test_labels = mnist["test_labels"]
