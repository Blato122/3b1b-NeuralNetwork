import random
import numpy as np
from typing import Final
import json
import utils


WIDTH: Final[int] = 28
HEIGHT: Final[int] = 28
INPUT_N: Final[int] = WIDTH * HEIGHT # 784
OUTPUT_N: Final[int] = 10

# ZAPISAC TO WSZYSTKO ZANIM ZAPOMNE!!! W README LUB KOMENTARZU!!!
# to z pochodna costu, to z suma po poprzednich layerach itd.

def load(name="my_network"):
    with open(f"../saves/{name}.json", "r") as f:
            data = json.load(f)
            cost = data["cost"]
            net = Network(data["sizes"], cost=cost) # does that work??
            net.weights = [np.array(w) for w in data["weights"]]
            net.biases = [np.array(b) for b in data["biases"]]
            return net

class Network:
    def save(self, name="my_network"):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": self.cost.__name__}

        with open(f"../saves/{name}.json", "w+") as f:
            json.dump(data, f)

    def __init__(self, sizes, cost=utils.SSECost):
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
        # and x is the number of neurons in the previous layer. We want to get a 2-dimensional
        # y by x array, containing x weights of connections from all x neurons
        # from the previous layer to the y-th neuron on the next layer.
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.cost = cost
    
    def feedforward_all(self, a):
        """Calculates zs and activations of all the neurons based on an input
        'a' vector."""
        activation = a
        activations = [a]
        zs = []
        for w, b in zip(self.weights, self.biases):
           z = np.dot(w, a) + b 
           zs.append(z) # vector of zs of one layer
           activation = utils.sigmoid(z)
           activations.append(activation) # vector of activations of one layer
           a = activation
        return (activations, zs)

    def feedforward_output(self, a):
        """Calculates the output of a neural network (i.e. activations
        of neurons is the last layer) based on an input 'a' vector"""
        for w, b in zip(self.weights, self.biases):
            a = utils.sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """Randomly shuffles the training data and divides it into mini_batches, each of mini_batch_size.
        Computes a gradient descent step using backprop according to the mini_batch.
        training_data - list of (x, y) tuples, where x is the training input and y is the desired output"""
        
        lr = learning_rate

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
            print(f"Before any learning: {self.evaluate(test_data)}/{n_test}")

        n = len(training_data)
        for i in range(epochs):
            # take random samples of the training data
            random.shuffle(training_data)
            mini_batches = [training_data[j:j+mini_batch_size] for j in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch_GD_backprop_step(mini_batch, lr)
            if test_data:
                print(f"Epoch {i+1} complete: {self.evaluate(test_data)}/{n_test}") # type: ignore
            else:
                print(f"Epoch {i+1} complete")
            # inverse square root schedule
            lr = learning_rate / np.sqrt(i+1)

    def update_mini_batch_GD_backprop_step(self, mini_batch, learning_rate):
            """Updates the network's weights and biases by applying gradient descent using backpropagation
            to a single mini_batch (gradient descent step)."""

            # same size as weight and biases and also DC_Dw_vect and DC_Db_vect
            w_step = [np.zeros(w.shape) for w in self.weights]
            b_step = [np.zeros(b.shape) for b in self.biases]
            for x, y in mini_batch:
                DC_Dw_vect, DC_Db_vect = self.backprop_compute_gradient(x, y)
                w_step = [w + dw for w, dw in zip(w_step, DC_Dw_vect)]
                b_step = [b + db for b, db in zip(b_step, DC_Db_vect)]
    # jak to sie dzieje ze sie wymiary zgadzaja??
            # minus because we have to move in the negative gradient direction
            # but we calculated (normal) gradient of the cost function
            # divided by the number of data instances in a mini_batch because
    # total cost function is an average of all costs (HUH?)
            self.weights = [w - (dw * (learning_rate/len(mini_batch))) for w, dw in zip(self.weights, w_step)] 
            self.biases = [b - (db * (learning_rate/len(mini_batch))) for b, db in zip(self.biases, b_step)]

    def backprop_compute_gradient(self, x, y):
        """backpropagation"""
        # same shape as weights and biases so that it's easy to
        # perform a gradient step, i.e. nudge all weights and
        # biases 
        DC_Dw_vect = [np.zeros(w.shape) for w in self.weights]
        DC_Db_vect = [np.zeros(b.shape) for b in self.biases]

        # feedforward
        activations, zs = self.feedforward_all(x)

        # chain rule derivatives

        # DC_Da = self.cost.DC_Da(activations[-1], y)
        # Da_Dz = sigmoid_d(zs[-1])
        # common_DC_factor = DC_Da * Da_Dz

        common_DC_factor = self.cost.DC_Da__Da_Dz(activations[-1], y, zs[-1])
        DC_Db_vect[-1] = common_DC_factor
        # why transpose?
        DC_Dw_vect[-1] = np.dot(common_DC_factor, activations[-2].transpose())


        # backward pass
        # begin with SECOND TO LAST layer and go backwards till layer 2 is reached
        for l in range(2, self.layer_num):
            z = zs[-l]
            Da_Dz_hidden = utils.sigmoid_d(z)
            # ogarnac wymiary!
            # i jakies transpozycje tutaj!?
            common_DC_factor = np.dot(self.weights[-l+1].transpose(), common_DC_factor) * Da_Dz_hidden
            DC_Db_vect[-l] = common_DC_factor # * 1 = Dz_Db
            DC_Dw_vect[-l] = np.dot(common_DC_factor, activations[-l-1].transpose())

        return DC_Dw_vect, DC_Db_vect

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward_output(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def classify_digit(self, pixels):
        results = self.feedforward_output(pixels)
        return (np.argmax(results), results)