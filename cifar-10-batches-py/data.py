# https://www.cs.toronto.edu/~kriz/cifar.html

import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main():
    # retrieve metadata about this dataset
    labels = unpickle('batches.meta')
    print(labels)

    # retrieve data on individual batch
    data_batch_1 = unpickle('data_batch_1')
    print(data_batch_1.keys())
    print(data_batch_1[b'data'].shape) # data and labels are numpy arrays

if __name__ == '__main__':
    main()
