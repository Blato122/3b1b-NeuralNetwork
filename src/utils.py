import numpy as np


def sigmoid(x):
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_d(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x) * (1 - sigmoid(x))



class SSECost:
    """Sum of squared errors."""
    @staticmethod
    def f(a, y):
        # return np.linalg.norm(a-y)**2 # ||a - y|| = sqrt(sum((ai - yi)^2))
        # return
        return np.sum(a - y)**2 # ?

    # @staticmethod
    # def DC_Da(a, y):
    #     return (a - y) * 2
    
    @staticmethod
    def DC_Da__Da_Dz(a, y, z):
        return (a - y) * sigmoid_d(z)

class CrossEntropyCost:
    """Cross entropy."""
    @staticmethod
    def f(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

    # @staticmethod
    # def DC_Da(a, y):
    #     pass
    
    @staticmethod
    def DC_Da__Da_Dz(a, y, z):
        return (a - y)

# class QuadraticCost(object):

#     @staticmethod
#     def fn(a, y):
#         """Return the cost associated with an output ``a`` and desired output
#         ``y``.

#         """
#         return 0.5*np.linalg.norm(a-y)**2

#     @staticmethod
#     def delta(z, a, y):
#         """Return the error delta from the output layer."""
#         return (a-y) * sigmoid_prime(z)


# class CrossEntropyCost:

#     @staticmethod
#     def fn(a, y):
#         """Return the cost associated with an output ``a`` and desired output
#         ``y``.  Note that np.nan_to_num is used to ensure numerical
#         stability.  In particular, if both ``a`` and ``y`` have a 1.0
#         in the same slot, then the expression (1-y)*np.log(1-a)
#         returns nan.  The np.nan_to_num ensures that that is converted
#         to the correct value (0.0).

#         """
#         return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

#     @staticmethod
#     def delta(z, a, y):
#         """Return the error delta from the output layer.  Note that the
#         parameter ``z`` is not used by the method.  It is included in
#         the method's parameters in order to make the interface
#         consistent with the delta method for other cost classes.

#         """
#         return (a-y)