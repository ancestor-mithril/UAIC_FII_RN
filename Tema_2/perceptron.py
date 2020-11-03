import multiprocessing
from typing import Tuple
import numpy as np
import pickle

from PIL import Image


def display_image(data, title):
    data = ((data - data.min()) / (data.max() - data.min())) * 255
    img = Image.fromarray(data.astype(np.uint8))
    # img.show(title=title)
    img.save(str(title) + ".png")


class Perceptron(multiprocessing.Process):
    def __init__(self, value: int = 0, niu: float = 0.01, iterations: int = 100, weights: np.ndarray = np.zeros(784),
                 bias: float = 0, one_hot_vector: list = None):
        super().__init__()
        if one_hot_vector is not None:
            self.value = one_hot_vector
            self.number = one_hot_vector.index(1)
        else:
            self.number = value
            self.value = np.array([1 if i == value else 0 for i in range(10)])
        self.training_set = None
        self.niu = niu
        self.iterations = iterations
        self.weights = weights
        self.bias = bias

    def run(self):
        print("Perceptron " + str(self.number) + " started training")
        if self.training_set is None:
            raise Exception("Training set not initialized")
        self.train_perceptron(self.training_set)
        self.to_file()

    def train_perceptron(self, training_set: Tuple[np.ndarray, np.ndarray]):
        """
        trains the perceptron using adeline perceptron

        :param training_set: tuple of 2 ndarrays, the former containing arrays of 784 pixels and the later the
        associated label with the real value
        :return: nothing
        """
        batch_len = 1000
        training_data = training_set[0]
        training_labels = np.array([[1 if x == i else 0 for i in range(10)] for x in training_set[1]])
        # print(training_labels)
        end_training = False
        while not end_training and self.iterations > 0:
            errors = 0
            for batch in range(int(training_data.shape[0] / batch_len)):
                delta_w = np.zeros(784)
                delta_b = 0
                for i in range(batch * batch_len, (batch + 1) * batch_len):
                    x = training_data[i]
                    predict = self.predict_value(x)
                    t = self.check_label(training_set[1][i])
                    # print("predict = ", predict, "t = ", t)
                    if t == 1 != predict:
                        errors += 1
                    # self.adjust(x, t - predict, self.niu)
                    difference = t - predict
                    delta_w = np.add(delta_w, np.multiply(x, difference * self.niu))
                    delta_b += difference * self.niu
                self.weights = np.add(self.weights, delta_w)
                self.bias += delta_b
            print(
                "Perceptron " + str(self.number) + " Iteration: " + str(self.iterations) + " , errors: " + str(errors)
            )
            self.iterations -= 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        """

        :param x: current training instance
        :return: current array prediction
        """
        return self.value * np.where((np.dot(x, self.weights) + self.bias) >= 0, 1, -1)

    def predict_value(self, x: np.ndarray) -> float:
        """

        :param x: current training instance
        :return: value of prediction -1 or 1
        """
        return np.where((np.dot(x, self.weights) + self.bias) >= 0, 1, -1)

    def predict_raw_value(self, x: np.ndarray) -> float:
        """

        :param x: current training instance
        :return: floating value of prediction
        """
        return np.dot(x, self.weights) + self.bias

    def check_label(self, label: int) -> int:
        """
        checks if current training label is equal to this perceptron

        :return:
        """
        return 1 if self.number == label else 0

    def adjust(self, x: np.ndarray, predict: float, niu: float = None):
        """

        :param niu: learning rate
        :param x: current training instance
        :param predict: prediction
        :return:
        """
        if niu is None:
            niu = self.niu
        self.weights = np.add(self.weights, np.multiply(x, predict * niu))
        self.bias += predict * self.niu
        # print(self.bias, type(self.bias))

    def to_file(self):
        """
        transforms the perceptron into a dictionary and saves it to file

        :return:
        """
        dictionary = {
            "value": self.value,
            "number": self.number,
            "niu": self.niu,
            "iterations": self.iterations,
            "weights": self.weights,
            "bias": self.bias,
        }
        with open(str(self.number) + '_perceptron.txt', 'wb') as file:
            pickle.dump(dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)

    def to_string(self):
        return str({
            "value": self.value,
            "number": self.number,
            "niu": self.niu,
            "iterations": self.iterations,
            "weights": self.weights,
            "bias": self.bias,
        }) + '\n'

    def to_image(self):
        pixels = np.reshape((self.weights + 1) / 2, [28, 28])
        display_image(pixels, str(self.number))

    def set_training_set(self, train_set: tuple):
        self.training_set = train_set
