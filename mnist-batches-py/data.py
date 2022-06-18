# sourced and modified from: https://github.com/hsjeong5/MNIST-for-Numpy
import pickle

class MNISTDataLoader():
    def mnist(self): # MNIST dataset
        with open("mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        self.train_images = mnist["training_images"]
        self.train_labels = mnist["training_labels"]
        self.test_images = mnist["test_images"]
        self.test_labels = mnist["test_labels"]
        self.name = "MNIST"
        return self

    def fashion_mnist(self): # fashion MNIST dataset
        with open("fashion_mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        self.train_images = mnist["training_images"]
        self.train_labels = mnist["training_labels"]
        self.test_images = mnist["test_images"]
        self.test_labels = mnist["test_labels"]
        self.name = "Fashion MNIST"
        return self

mnist = MNISTDataLoader().mnist()
fashion_mnist = MNISTDataLoader().fashion_mnist()
