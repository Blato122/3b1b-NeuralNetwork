import network

if __name__ == "__main__":
    # data...
    network = network.Network([784, 12, 12, 12, 10])
    network.SGD(..., 100, 100, 2.0)