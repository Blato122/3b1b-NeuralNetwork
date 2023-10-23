import numpy as np


class Sigmoid:
    @staticmethod
    def f(x):
        """Sigmoid function."""
        return 1.0 / (1.0 + np.exp(-x))
    
    @staticmethod
    def Da_Dz(x):
        """Derivative of the sigmoid function."""
        return Sigmoid.f(x) * (1 - Sigmoid.f(x))
    
class Softmax:
    @staticmethod
    def f(x):
        """Softmax function."""
        exps = np.exp(x - x.max()) # -x.max() for "numerical stability"
        return exps / np.sum(exps)
    
    @staticmethod
    def Da_Dz(x):
        """Derivative of the softmax function."""
        # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        # x = x.reshape(-1,1)
        # return np.diagflat(x) - np.dot(x, x.T)
        pass