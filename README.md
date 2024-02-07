# Customizable Neural Network for MNIST Digit Classification

## Project Overview

    This project features a customizable, fully-connected neural network implemented from scratch in Python, designed for the classification of handwritten digits from the MNIST dataset. Unlike traditional, single-architecture models, this implementation provides a flexible framework through the Layer and NN classes, allowing for the creation of neural networks with any number of layers and neurons, and supports a number of activation functions.
    This implementation from scratch supports forward propagation, backpropagation, and gradient descent optimization.

## Features

    Modular Design: Utilizes Layer and NN classes within model.py to facilitate the construction of neural networks with varying architectures.
    Data Preprocessing: Includes normalization and vectorization of the MNIST dataset to prepare it for training.
    Optimization: Utilizes gradient descent for parameter optimization with a focus on numerical stability in softmax calculations.
    Batch Processing: Supports mini-batch processing for improved training performance and efficiency.

## Architecture Customization

The project allows for significant customization in network architecture:

    Dynamic Layer Configuration: Users can easily define networks with varying sizes and depths by specifying layers in the NN class constructor.
    Activation Function Flexibility: Each layer can independently utilize different activation functions, including ReLU, sigmoid, tanh, and softmax.
    Example Configuration: The provided example includes a network with two hidden layers, each with 512 neurons, demonstrating the model's capability to handle complex architectures.

## Results

    After 10 epochs of training with a learning rate of 1e-3, the model achieved an accuracy of approximately 92.33% on the MNIST test set. Suggestions for further improvements and potential modifications for achieving higher accuracy are discussed.

## Technologies Used

    Python
    NumPy
    Matplotlib (for plotting training progress)

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

## Potential Improvements

    Experimentation with different network architectures and hyperparameters.
    Implementation of regularization techniques such as dropout or L2 regularization to combat overfitting.
    Use of convolutional neural network (CNN) layers for potentially improved performance on image-based tasks.

## Authors

    Wajdi Maatouk

## Acknowledgments

    Inspiration and guidance from the Neural Networks and Deep Learning course by DeepLearning.AI, available on Coursera. 
    MNIST dataset provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.