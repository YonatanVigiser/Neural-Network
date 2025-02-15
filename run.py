import DrawingPad
from NeuralNetwork import NeuralNetwork
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
# import matplotlib.pyplot as plt

network = NeuralNetwork([784, 64, 32, 10])
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()


def load_test_image(image_number):
    arr = np.array(test_images[image_number])
    arr = arr.tolist()
    res = []

    for n in arr:
        for x in n:
            res.append(x / 255)

    return res


def load_train_image(image_number):
    arr = np.array(train_images[image_number])
    arr = arr.tolist()
    res = []

    for n in arr:
        for x in n:
            res.append(x / 255)

    return res


def paint():
    DrawingPad.DrawingPad.run()
    digit = input("\nPlease enter the digit you have written: ")
    while not digit.isdigit():
        print("Please enter a digit! Try again.")
        digit = input("Please enter the digit you have written: ")

    drawn_image = Image.open('digit.png')
    drawn_image = drawn_image.resize((28, 28))
    drawn_image = drawn_image.convert('L')
    drawn_image = ImageOps.invert(drawn_image)
    drawn_image = drawn_image.filter(ImageFilter.GaussianBlur(radius=1))
    array = np.array(drawn_image) / 255.0
    array = array.reshape((1, 28, 28, 1))
    array = array.flatten().tolist()

    # To display the imagee drawn or any other image:
    # plt.imshow(image, cmap='gray')
    # plt.title(f"Label: {digit}")
    # plt.axis('off')
    # plt.show()

    result = network.run_network(array)

    if result[-1].index(max(result[-1])) == int(digit):
        print(f"\nGuessed correctly! ({round(max(result[-1]) * 100, 2)}% confidence)")
    else:
        print(f"\nGuessed incorrectly ({result[-1].index(max(result[-1]))})! "
              f"({round(max(result[-1]) * 100, 2)}% confidence)")
    again = input("\nDo you want to draw again? (Y/n): ")
    if again.upper() == "Y":
        paint()


training_count = 60000
testing_count = 10000
learning_rate = 0.1


train_images_result = []

for i in range(training_count):
    image = load_train_image(i)
    train_images_result.append(image)

test_images_result = []

for i in range(testing_count):
    image = load_test_image(i)
    test_images_result.append(image)

train = input("Do you want to use an exsisting saved model? (Y/n): ")
print()

if train.upper() == "Y":
    weights_file = open("weights.txt", "r")
    network.weights = eval(weights_file.read())
    weights_file.close()
    biases_file = open("biases.txt", "r")
    network.biases = eval(biases_file.read())
    biases_file.close()


else:
    network.train(train_images_result, train_labels[:training_count], learning_rate)

do_paint = input("Do you want to to paint the number? (Y/n): ")
if do_paint.upper() == "Y":
    paint()
else:
    network.test(test_images_result, test_labels[:testing_count])


if train.upper() != "Y":
    save = input("\nDo you want to save the network values? (Y/n): ")
    if save.upper() == "Y":
        weights_file = open("weights.txt", "w")
        weights_file.write(str(network.weights))
        weights_file.close()
        biases_file = open("biases.txt", "w")
        biases_file.write(str(network.biases))
        biases_file.close()
