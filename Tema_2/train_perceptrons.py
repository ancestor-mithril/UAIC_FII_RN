import pickle
from typing import List, Tuple
import numpy as np

from perceptron import Perceptron
from test_perceptrons import test_perceptrons


def train_perceptrons_together(perceptrons: List[Perceptron], training_set: Tuple[np.ndarray, np.ndarray],
                               test_set: Tuple[np.ndarray, np.ndarray], iterations: int = 10, niu: float = 0.01):
    """

    :param niu: learning rate
    :param iterations: number of iterations to be made
    :param perceptrons: the perceptrons to be trained
    :param training_set: tuple of 2 ndarrays, the former containing arrays of 784 pixels and the later the
    associated label with the real value
    :param test_set: --//--
    :return: nothing
    """
    training_data = training_set[0]
    training_labels = np.array([[1 if x == i else 0 for i in range(10)] for x in training_set[1]])
    end_training = False
    perceptrons_wights = np.array([x.weights for x in perceptrons], dtype="float64")
    perceptron_biases = np.array([x.bias for x in perceptrons], dtype="float64")
    print(perceptrons_wights)
    while not end_training and iterations > 0:
        errors = 0
        for i in range(training_data.shape[0]):
            x = np.array([training_data[i]])
            z = np.add(np.dot(perceptrons_wights, x[0]), perceptron_biases)  # adeline activation
            print(z)
            diff = np.array([training_labels[i] - z])
            if diff[0, np.amax(training_labels[i])] != 0:
                errors += 1
            a = (x * diff.transpose()) * niu
            perceptrons_wights += a
            perceptron_biases += diff[0] * niu

        print(" Iteration: " + str(iterations) + " , errors: " + str(errors))
        iterations -= 1

    for x in range(perceptrons_wights.shape[0]):
        perceptrons[x].weights = perceptrons_wights[x]
        perceptrons[x].bias = perceptron_biases[x]

    print(test_perceptrons(perceptrons, test_set))


def train_perceptrons_subprocess(training_set: Tuple[np.ndarray, np.ndarray], test_set: Tuple[np.ndarray, np.ndarray],
                                 iterations: int = 10, niu: float = 0.01, ):
    """

    :param training_set: tuple of 2 ndarrays, the former containing arrays of 784 pixels and the later the
    associated label with the real value
    :param test_set: --//--
    :param niu: learning rate
    :param iterations: number of iterations to be made
    :return: nothing
    """
    perceptrons = [Perceptron(value=i, iterations=iterations, niu=niu) for i in range(10)]
    for p in perceptrons:
        p.set_training_set(training_set)
        p.start()
    for p in perceptrons:
        p.join()
    perceptrons = []
    for i in range(10):
        with open(str(i) + "_perceptron.txt", 'rb') as file:
            x = pickle.load(file)
        perceptrons.append(Perceptron(value=x['number'], niu=x['niu'], iterations=x['iterations'], weights=x['weights'],
                                      bias=x['bias']))
    print(test_perceptrons(perceptrons, test_set))
