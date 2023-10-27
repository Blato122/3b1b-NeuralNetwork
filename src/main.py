import network as n
import mnist
import image
import numpy as np
import utils


if __name__ == "__main__":
    training_data, validation_data, test_data = mnist.load()
    training_data = list(training_data)
    test_data = list(test_data)
    net = n.Network([n.INPUT_N, 50, n.OUTPUT_N], cost=utils.CrossEntropyCost) # type: ignore
    net.SGD(training_data, epochs=80, mini_batch_size=20, learning_rate=5.0, test_data=test_data)
    input("Learning finished. Press any key to start testing")

    print("\nTesting:\n")
    offset = 10
    num = 10
    digits = [x for (x, y) in test_data[offset:offset+num]]
    digits.append(image.image_to_grayscale("my_digit_2"))
    digits.append(image.image_to_grayscale("my_digit_3"))
    for i, digit_img in enumerate(digits):
        name = f"img{i}"
        print(name)
        image.create_image(digit_img, n.WIDTH, n.HEIGHT, name)
        image.display_image_terminal(name, 2 * n.WIDTH)
        digit_guess, activations = net.classify_digit(digit_img)
        print(f"I think img{i} is a(n) {digit_guess}:\n {activations}")
        input("Press any key to continue\n")

    # net.save()

# to do:
# cost function ogarnac
# opis w klasie network, dokladnie bardzo, zebym za pol roku tez ogarnial

