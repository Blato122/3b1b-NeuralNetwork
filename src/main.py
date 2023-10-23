import network as n
import mnist
import image
import numpy as np
import utils


if __name__ == "__main__":
    training_data, validation_data, test_data = mnist.load()
    training_data = list(training_data)
    test_data = list(test_data)
    net = n.Network([n.INPUT_N, 80, n.OUTPUT_N], cost=utils.CrossEntropyCost)
    net.SGD(training_data, epochs=5, mini_batch_size=20, learning_rate=3.0, test_data=test_data) #xentr parameter
    input("Learning finished. Press any key to start testing")

    print("\nTesting:\n")
    digits = [x for (x, y) in test_data[:3]]
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

# to do:

# adjustowac jakos learning rate?? zmniejszac

# dlaczego aktywacje maja takie dziwne wyniki?

# cost function ogarnac

# opis w klasie network, dokladnie bardzo, zebym za pol roku tez ogarnial

# po zakonczeniu nauki niech sie zapyta czy chce zapisac i jesli tak
# to zapisuje weights and biases itd zeby moc potem zaladowac z pliku juz nauczona

# cross entropy, softmax