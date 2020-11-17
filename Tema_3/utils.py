import json

import numpy as np
from PIL import Image


def unison_shuffled_copies(a, b):
    """
    https://stackoverflow.com/a/4602224

    :param a:
    :param b:
    :return:
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def apply_sigmoid(x: np.ndarray) -> np.ndarray:
    """

    :param x: np array of neuron predictions, should be (x, 1), with x in {100, 10}
    :return: array of activated predictions
    """
    return 1.0 / (1.0 + np.exp(-x))


def soft_max(x: np.ndarray) -> np.ndarray:
    """

    :param x: last layer array of predictions, should be (10, 1)
    :return: soft max activation
    """
    x = np.exp(x)
    return np.divide(x, np.sum(x))


def get_sigmoid_prime(x: np.ndarray):
    """

    :param x: np . array of predictions
    :return: derivative of sigmoid
    """
    return apply_sigmoid(x) * (1 - apply_sigmoid(x))


def to_image(data: np.ndarray):
    """
    shows image before and after blurring

    :param data: [28, 28]
    :return:
    """
    dx = [-1, -1, -1, 0, 1, 1, 1, 0]
    dy = [-1, 0, 1, 1, 1, 0, -1, -1]

    img = Image.fromarray(np.subtract(255, np.multiply(data, 255)).astype(np.uint8))
    img.show()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            s = 0
            for ii, jj in zip(dx, dy):
                x = i + ii
                y = j + jj
                if 0 <= x < data.shape[0] and 0 <= y < data.shape[1]:
                    s += data[x, y]
            s /= 8
            data[i, j] = s
    data = np.subtract(255, np.multiply(data, 255))
    img = Image.fromarray(data.astype(np.uint8))
    img.show()


def blur_image(data: np.ndarray):
    """

    :param data:
    :return: blurred image
    """
    copy_data = np.array(data, copy=True)
    dx = [-1, -1, -1, 0, 1, 1, 1, 0]
    dy = [-1, 0, 1, 1, 1, 0, -1, -1]
    for i in range(copy_data.shape[0]):
        for j in range(copy_data.shape[1]):
            s = 0
            for ii, jj in zip(dx, dy):
                x = i + ii
                y = j + jj
                if 0 <= x < copy_data.shape[0] and 0 <= y < copy_data.shape[1]:
                    s += copy_data[x, y]
            s /= 8
            copy_data[i, j] = s
    return copy_data


def generate_more_training_data(data):
    """

    :param data:
    :return:
    """
    blurred_set = []
    for i in range(data[0].shape[0]):
        blurred_set.append(blur_image(data[0][i].reshape([28, 28])).reshape((784,)))

    blurred_set = np.array(blurred_set)

    tt = np.concatenate((data[0], blurred_set), axis=0)
    tt2 = np.concatenate((data[1], data[1]), axis=0)
    return tt, tt2
