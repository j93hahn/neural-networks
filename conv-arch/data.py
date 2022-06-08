# sourced and modified from: https://github.com/hsjeong5/MNIST-for-Numpy
import pickle

class DataLoader():
    def __init__(self) -> None:
        with open("mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        self.x_train = mnist["training_images"].reshape(60000, 28, 28)
        self.t_train = mnist["training_labels"]
        self.x_test = mnist["test_images"].reshape(10000, 28, 28)
        self.t_test = mnist["test_labels"]
