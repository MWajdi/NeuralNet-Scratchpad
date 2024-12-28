# Customizable Neural Network for MNIST Digit Classification

## Project Overview

This project features a customizable, fully-connected neural network implemented from scratch in Python. It is then used for the classification of handwritten digits from the MNIST dataset. This implementation provides a flexible framework through the Layer and NN classes, allowing for the creation of neural networks with any number of layers and neurons, and supports a number of activation functions.
It supports forward propagation, backpropagation, and gradient descent optimization.


## Results

After 10 epochs of training with a learning rate of 1e-3, the model achieved an accuracy of approximately 92.33% on the MNIST test set. Suggestions for further improvements and potential modifications for achieving higher accuracy are discussed.


## Getting Started
### Prerequisites

    Python 3.x
    NumPy
    Matplotlib

### Running the Model

    Clone the repository.
    Install the required Python packages.
    Execute the script: python modular_implementation.py.

### Directory Structure

    model.py: Contains the Layer and NN classes for building customizable neural networks.
    modular_implementation.py: Demonstrates how to create and train a model using the custom neural network framework.
    data_preparation.py: Defines the Dataset class for loading and preprocessing the MNIST dataset.
    simple_implementation.py: Provides a simpler, non-class-based implementation for comparison.


## Acknowledgments

    Inspiration and guidance from the Neural Networks and Deep Learning course by DeepLearning.AI, available on Coursera. 
    MNIST dataset provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
