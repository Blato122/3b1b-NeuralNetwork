import network
import mnist
import image
import numpy as np

if __name__ == "__main__":
    training_data, validation_data, test_data = mnist.load()
    training_data = list(training_data)
    test_data = list(test_data)
    net = network.Network([784, 40, 10])
    net.SGD(training_data, 5, 20, 2.0, test_data=test_data)
    input("Learning finished. Press any key to start testing")

    print("\nTesting:\n")
    digits = [x for (x, y) in test_data[:10]]
    for i, digit_img in enumerate(digits):
        name = f"img{i}"
        print(name)
        image.create_image(digit_img, 28, 28, name)
        image.display_image_terminal(name, 56)
        digit_guess = net.classify_digit(digit_img)
        print(f"I think img{i} is a {digit_guess}")
        input("Press any key to continue\n")

    pixels = image.image_to_grayscale("../data/MNIST_6_0.png")
    image.display_image_terminal("MNIST_6_0", 56)
    guess = net.classify_digit(pixels)
    print(f"I think it is a {guess}")




        
# to do:

# po zakonczeniu nauki niech sie zapyta czy chce zapisac i jesli tak
# to zapisuje weights and biases itd zeby moc potem zaladowac z pliku juz nauczona

# dodac 784 i 28 jako jakies zmienne stale globalne
