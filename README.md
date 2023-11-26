# neural-networks
Source code for my personal project focused on building a vectorized neural network library. A presentation can be found here: [https://docs.google.com/presentation/d/1fM2uixEnTXKnPIQLfQKK7edWO8Xq4ySZWtMglCW9cMY/edit#slide=id.p](https://docs.google.com/presentation/d/1fM2uixEnTXKnPIQLfQKK7edWO8Xq4ySZWtMglCW9cMY/edit?usp=sharing)

### modules/

The modules/ directory is structured to mirror PyTorch's torch.nn, with the goal being concise and easy-to-read code. I've developed the code for all of these models using only NumPy in order to solidify my own understanding of the math behind back-propagation and develop fluency with neural network experiments. Models currently include the multi-layer perceptron (a.k.a, vanilla network) and a 2D convolutional neural network (for image classification). The code contains fully vectorized code using NumPy's built-in C implementations, so that I can learn advanced indexing and become fluent with NumPy's API, which is very similar to PyTorch's.

### optim/

The optim/ directory is structured to mirror PyTorch's torch.optim library. I currently use optim/ to store the code for my optimizers and learning rate schedulers. Examples of optimizers include SGDM and Adam.

### data_loaders/

The data_loaders/ directory contains the data I am using to train my models upon. These include the MNIST and FashionMNIST. Stored elsewhere is the CIFAR-10 dataset under the conv/ folder. These are all tailored for image classifications tasks and thus are optimal datasets for training my models upon.

### plots/

The plots/ directory stores all plots and metadata related to said plots. These plots display a variety of things from error on training sets to experimental findings and more. Feel free to take a look!

### test_framework/

This directory is used to check the numeric gradient calculations of my various modular layers, designed to ensure that the backpropagation calculus is working correctly and that the model is actually learning.

### mlp/

The mlp/ directory houses specific information about dozens of multi-layer perceptron models I've trained (whose training specifications you can find in the metadata.json file stored on that level of my repository). Some things stored are the models themselves (stored via torch.save) and landmark papers detailing these models' infrastructure and design. You can also find information on various experiments I've run with my various model specifications, helping me discover new information about different weight initialization and normalization techniques.

### conv/

The conv/ directory is the jewel of the crown for my entire neural-networks/ repository. This repository contains various testing files, designed to compare my implementation of a 2d convolutional layer against PyTorch's own code (which I match), showing that my code does indeed work and that the numerical gradient calculations are correct. You can also find other files showing me playing around with PyTorch's API in order to get a better sense of the torch tensor and its similarites to the NumPy ndarray.

### diffusion-arch/

This directory contains several papers on diffusion modeling, as well as a link to a diffusion model hackathon which I participated in with members of my research lab in June 2022.

### results/

This subdirectory contains information on the experiments I ran, as well as the script I used to create my results. As part of these experiments, I also used a Jupyter notebook to visualize the gradients and parameters of various models in order to play around and uncover relationships behind weight initialization and weight normalization.
