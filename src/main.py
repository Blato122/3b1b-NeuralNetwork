import network
import mnist

if __name__ == "__main__":
    training_data, validation_data, test_data = mnist.load()
    training_data = list(training_data)
    network = network.Network([784, 12, 12, 12, 10])
    network.SGD_learn(training_data, 100, 100, 2.0)