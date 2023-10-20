import random
import numpy as np

class Network:

    def __init__(self, sizes):
        """Sets up:
        layer_num - number of layers of a network.
        sizes - list of sizes of a network's layers.
        weights - list containing arrays of weights between every two layers.
            Each element of the list contains weights connecting every neuron 
            in the previous layer with each neuron in the next layer.
            Example: 
                layer1 - 5 neurons
                layer2 - 3 neurons
            Each neuron from layer2 is connected to all 5 neurons from 
            the previous layer1. That means, we have to store all these 3*5 = 15
            weights in an 3x5 array. Next list element will store weights 
            between layer2 and layer3 etc.
        biases - list containing biases in each layer except for the first
            one (the input layer).
        """
        self.layer_num = len(sizes)
        self.sizes = sizes
        # randn() -> generates samples from the Standard Normal (aka. Gaussian) distribution (mean 0 and variance 1)
        # y, x -> dimensions of the array. y is the number of neurons in the next layer
        # and x is the number of neurons in the previous layer. We want to get a y-dimensional
        # array, each dimension containing x weights of connections from all x neurons
        # from the previous layer to the y-th neuron on the next layer.
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    
    def feedforward(self, activation):
        """Calculates the output of a neural network based on an input 'activation' vector"""
        for w, b in zip(self.weights, self.biases):
           activation = sigmoid(np.dot(w, activation) + b) # ogarnac jak to sie dzieje, ze wymiar sie zgadza
        return activation
    
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """Randomly shuffles the training data and divides it into mini_batches, each of mini_batch_size.
        Computes a gradient descent step using backprop according to the mini_batch.
        training_data - list of (x, y) tuples, where x is the training input and y is the desired output"""
        #...
        #...
        n = len(training_data)
        for i in range(epochs):
            # take random samples of the training data
            random.shuffle(training_data)
            mini_batches = [training_data[j:j+mini_batch_size] for j in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch_GD_step(mini_batch, learning_rate)
            print(f"Epoch {i+1} complete")

    def update_mini_batch_GD_step(self, mini_batch, learning_rate): # learning rate ogarnac gdzie sie uzywa dokladnie tutaj
        """Updates the network's weights and biases by applying gradient descent using backpropagation
        to a single mini_batch (gradient descent step)."""
        for x, y in mini_batch:
            pass
        pass

    def backprop(self, x, y):
        """backpropagation"""
        weight_derivatives = [np.zeros(w.shape) for w in self.weights]
        bias_derivatives = [np.zeros(b.shape) for b in self.biases]

        # feedforward
        # nie lepiej dac to do funkcji feedforward?!?!
        activation = x
        activations = [x] # list representing all the activation vectors layer by layer
        zs = [] # list representing all z vectors (activations before applying sigmoid) layer by layer
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        for layer in range(self.layer_num, 2, -1):
            z = zs[layer]
            a = sigmoid_d(z)
            # przeczytac tekst i ogarnac bo tam jest inaczej niz na filmie!


def sigmoid(x):
    """The sigmoid function"""
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_d(x):
    """Derivative of the sigmoid function"""
    return sigmoid(x) * (1 - sigmoid(x))