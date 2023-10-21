import random
import numpy as np

"""make this more in OOP style!
layer class maybe
gradient function
neuron as a class also???
zestaw 5 python klasy moze pomoze"""

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
        # klasa layer potem? a moze nawet neuron ale nwm
        self.layer_num = len(sizes)
        self.sizes = sizes
        # randn() -> generates samples from the Standard Normal (aka. Gaussian) distribution (mean 0 and variance 1)
        # y, x -> dimensions of the array. y is the number of neurons in the next layer
        # and x is the number of neurons in the previous layer. We want to get a 2-dimensional
        # y by x array, containing x weights of connections from all x neurons
        # from the previous layer to the y-th neuron on the next layer.
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        
        self.zs = []
        self.activations = []
    
    def feedforward_update(self, a):
        """Calculates the output of a neural network based on an input 'a' vector
        Updates all zs and activations calculated during a forward pass"""
        # each loop iteration updates zs and activations in one layer
        for w, b in zip(self.weights, self.biases):
           z = np.dot(w, a) + b 
           self.zs.append(z) # vector of zs of one layer
           activation = sigmoid(z)
           self.activations.append(activation) # vector of activations of one layer
           a = activation
        return (self.zs, self.activations)

    def feedforward_output(self, a):
        """ Only calculates the output of a neural network based on an input 'a' vector"""
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD_learn(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """Randomly shuffles the training data and divides it into mini_batches, each of mini_batch_size.
        Computes a gradient descent step using backprop according to the mini_batch.
        training_data - list of (x, y) tuples, where x is the training input and y is the desired output"""
        #... test data
        #...
        n = len(training_data)
        for i in range(epochs):
            # take random samples of the training data
            random.shuffle(training_data)
            mini_batches = [training_data[j:j+mini_batch_size] for j in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch_GD_backprop_step(mini_batch, learning_rate)
            print(f"Epoch {i+1} complete")

    def update_mini_batch_GD_backprop_step(self, mini_batch, learning_rate):
            """Updates the network's weights and biases by applying gradient descent using backpropagation
            to a single mini_batch (gradient descent step)."""

            w_step = [np.zeros(w.shape) for w in self.weights]
            b_step = [np.zeros(b.shape) for b in self.biases]
            for x, y in mini_batch:
                DC_Dw_vect, DC_Db_vect = self.backprop_compute_gradient(x, y) # zla nazwa, to juz sa wektory liczb!!?
                w_step = [w + dw for w, dw in zip(w_step, DC_Dw_vect)]
                b_step = [b + db for b, db in zip(b_step, DC_Db_vect)]
            # jak to sie dzieje ze sie wymiary zgadzaja??
            self.weights = [w - dw * learning_rate/len(mini_batch) for w, dw in zip(self.weights, w_step)] 
            self.biases = [b - db * learning_rate/len(mini_batch) for b, db in zip(self.biases, b_step)]

    def backprop_compute_gradient(self, x, y):
        """backpropagation"""
        # same shape as weights and biases so that its easy to
        # perform a gradient step, i.e. nudge all weights and
        # biases 
        DC_Dw_vect = [np.zeros(w.shape) for w in self.weights]
        DC_Db_vect = [np.zeros(b.shape) for b in self.biases]

        # feedforward, updates zs and activations
        # ale pomieszane, jeszcze appendowanie tego x...
        self.activations.append(x)
        self.feedforward_update(x)

        # co z costem? nigdzie sie tego nie uzywa? ale no jak...
        # OK - uzywa sie przeciez!!! jako DC_Da!
        # cost = (self.activations[-1] - y)**2

        # pochodne z chain rule
        # tez poogaraniac te wymiary jak to sie udaje XD
        # COST DERIVATIVE!!!
        DC_Da = 2 * (self.activations[-1] - y)
        Da_Dz = sigmoid_d(self.zs[-1])

        common_DC_factor = Da_Dz * DC_Da
        DC_Db_vect[-1] = common_DC_factor
        DC_Dw_vect[-1] = np.dot(self.activations[-2], common_DC_factor) # transpozycja tutaj?


        # backward pass
        # begin with SECOND TO LAST layer and go backwards till layer 2 is reached
        for l in range(2, self.layer_num):
            z = self.zs[-l]
            Da_Dz_hidden = sigmoid_d(z)
            # ogarnac wymiary!
            # i jakies transpozycje tutaj!?
            common_DC_factor = np.dot(self.weights[-1], common_DC_factor) * Da_Dz_hidden
            DC_Dw_vect[-l] = np.dot(self.activations[-l-1], common_DC_factor)
            DC_Db_vect[-l] = common_DC_factor # * 1 = Dz_Db

        return DC_Dw_vect, DC_Db_vect

    def evaluate(self, test_data):
        # zmienic feedforward na feedforward ale taki bez modyfikacji
        test_results = [(np.argmax(self.feedforward_output(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

            
def sigmoid(x):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_d(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x) * (1 - sigmoid(x))