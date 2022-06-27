# neural-networks
Source code for all of my personal projects focused on building neural networks

### modules

The modules/ directory is structured to mirror PyTorch's torch.nn, with the goal being concise and easy-to-read code. I've developed the code for all of these models using only NumPy in order to solidify my own understanding of the math behind back-propagation and develop fluency with neural network experiments. Models currently include the multi-layer perceptron (a.k.a, feed-forward network) and a 2D convolutional neural network (for image classification).

### data_loaders

The data_loaders/ directory contains the data I am using to train my models upon. These include the MNIST, FashionMNIST, and Cifar-10 datasets. These are all tailored for image classifications tasks and thus are optimal datasets for training my models upon.

### plots

The plots/ directory stores all plots and metadata related to said plots. These plots display a variety of things from error on training sets to experimental findings and more. Feel free to take a look!

### *-arch

The *-arch libraries house specific information about each of the models I am training (whose training specifications you can find in the corresponding *-training.py file stored on this level of my repository). Some things stored are the models themselves (stored via torch.save) and landmark papers detailing these models' infrastructure and design.
