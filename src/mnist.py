import gzip
import pickle
import numpy as np
import network as n


"""data consists of..."""

def load_internal():
    # rb - read binary mode
    with gzip.open("../data/mnist.pkl.gz", "rb") as mnist_data:
        # encoding="latin1" is needed for numpy arrays
        training_data, validation_data, test_data = pickle.load(mnist_data, encoding="latin1")
        return (training_data, validation_data, test_data)

def load():
    tr_d, va_d, te_d = load_internal()

    # creates a list of 2d, 784 by 1 numpy arrays 
    training_inputs = [np.reshape(x, (n.INPUT_N, 1)) for x in tr_d[0]]
    # creates a list of 2d, 10 by 1 numpy arrays
    training_outputs = [output_to_vector(y) for y in tr_d[1]]
    # creates tuples each containing training inputs and corresponding outputs
    training_data = zip(training_inputs, training_outputs)

    # in case of validation and test data sets, we dont need to convert outputs
    # to vectors, because we won't use this data to learn - a simple number is enough
    validation_inputs = [np.reshape(x, (n.INPUT_N, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (n.INPUT_N, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return (training_data, validation_data, test_data)

def output_to_vector(idx):
    v = np.zeros((n.OUTPUT_N, 1))
    v[idx] = 1.0
    return v