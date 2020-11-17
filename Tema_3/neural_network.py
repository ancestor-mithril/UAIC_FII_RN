from typing import List
import numpy as np
import pickle

from utils import unison_shuffled_copies, apply_sigmoid, soft_max, get_sigmoid_prime


class Network(object):
    def __init__(self, layers: List[int], weights: List[np.ndarray] = None, biases: List[np.ndarray] = None):
        """

        :param layers: list containing the number of neurons on each layer
        """
        self.dropout_probability = 0.1
        self.layers = layers
        if weights is None:
            self.weights = self.weight_initialization()
        else:
            self.weights = weights
        if biases is None:
            self.biases = self.bias_initialization()
        else:
            self.biases = biases

    def weight_initialization(self) -> List[np.ndarray]:
        """
        initialization is done by generating random numbers with the mean=loc=0, and sd=scale=sqrt(weights that enter
        the neuron), and size is a matrix having all weights for each neuron on the same layer

        :return: a list of matrices of weights for each intermediary layer
        """
        return [
            np.random.normal(loc=0, scale=(1 / (np.sqrt((self.layers[i])))), size=(self.layers[i + 1], self.layers[i]))
            for i in range(len(self.layers) - 1)
        ]

    def bias_initialization(self) -> List[np.ndarray]:
        """

        :return: list of biases for each neuron on each layer
        """
        return [
            np.random.normal(loc=0, scale=(1 / (np.sqrt((self.layers[i])))), size=(self.layers[i + 1], 1))
            for i in range(len(self.layers) - 1)
        ]

    def train_neural_network(
            self, training_set: (np.ndarray, np.ndarray), iterations: int = 30, learning_rate: float = 0.01,
            mini_batch_size: int = 10, testing_set: (np.ndarray, np.ndarray) = None, apply_dropout: bool = False,
            dropout_rate: float = 0.1
    ):
        """

        :param training_set:
        :param iterations:
        :param learning_rate:
        :param mini_batch_size:
        :param testing_set: if None, then it is training set, to measure training accuracy
        :param apply_dropout: If True, applies dropout
        :param dropout_rate: should be between 0 and 1
        :return: noting
        """
        self.dropout_probability = dropout_rate
        if testing_set is None:
            testing_set = training_set
        training_data = training_set[0]
        training_labels = np.array([[[1] if x == i else [0] for i in range(10)] for x in training_set[1]])
        testing_data = testing_set[0]
        testing_labels = np.array([[[1] if x == i else [0] for i in range(10)] for x in testing_set[1]])
        end_training = False
        dropout = None
        while not end_training and iterations > 0:
            iterations -= 1
            training_data, training_labels = unison_shuffled_copies(training_data, training_labels)
            mini_batches = [
                (training_data[x:x + mini_batch_size], training_labels[x:x + mini_batch_size])
                for x in range(0, training_labels.shape[0], mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.train_mini_batch(mini_batch, learning_rate, dropout=dropout, apply_dropout=apply_dropout)

            print("Iteration: ", iterations, "training accuracy: ", self.evaluate(testing_data, testing_labels))

        self.save_to_file()

    def train_mini_batch(self, training_data: (np.ndarray, np.ndarray), learning_rate: float,
                         dropout: List[np.ndarray] = None, apply_dropout: bool = False):
        """
        we generate the dropout neurons once per mini batch, so that the weights are modified consistently per
        mini batch

        :param training_data: np array of data
        :param learning_rate: np array of correct labels
        :param dropout: dropout list for this iteration
        :param apply_dropout:
        :return: nothing
        """
        if apply_dropout:
            dropout = [
                np.random.choice([1, 0], (i.shape[0], 1),
                                 p=[1 - self.dropout_probability, self.dropout_probability])
                for i in self.weights
            ]
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        delta_biases = [np.zeros(b.shape) for b in self.biases]
        for i in range(training_data[0].shape[0]):
            d_w, d_b = self.train_instance(training_data[0][i], training_data[1][i], dropout=dropout,
                                           apply_dropout=apply_dropout)
            delta_weights = [d_w[i] + delta_weights[i] for i in range(len(delta_weights))]
            delta_biases = [d_b[i] + delta_biases[i] for i in range(len(delta_biases))]
        self.weights = [self.weights[i] - (delta_weights[i] * learning_rate) for i in range(len(delta_weights))]
        self.biases = [self.biases[i] - (delta_biases[i] * learning_rate) for i in range(len(delta_biases))]

    def train_instance(self, training_data: np.ndarray, training_label: np.ndarray, dropout: List[np.ndarray],
                       apply_dropout: bool = False) -> (List[np.ndarray], List[np.ndarray]):
        """
        using feed_forward we get the raw and activated values obtained by our network
        then we back propagate the error and return the modifications to be made

        :param training_data: a single instance of training data, an array of 784 elements
        :param training_label: [[0] [1] ... [0]] - one hot matrix?
        :param dropout: dropout list for this iteration
        :param apply_dropout: if True, then dropout should be applied
        :return: ([], []), delta weights and delta biases, the modifications to be made after each mini-batch
        """
        raw_z_values, activated_z_values = self.feed_forward(training_data, dropout=dropout,
                                                             apply_dropout=apply_dropout)
        return self.back_propagation(training_label, raw_z_values, activated_z_values)

    def back_propagation(self, training_label: np.ndarray, raw_z_values: List[np.ndarray],
                         activated_z_values: List[np.ndarray]) -> (List[np.ndarray], List[np.ndarray]):
        """
        delta is the difference between the network prediction (activated values) and real training_label,
        divided by derivative of sigmoid

        :param training_label: one hot matrix
        :param raw_z_values:
        :param activated_z_values:
        :return: ([], []), delta weights and delta biases, the modifications to be made after each mini-batch
        """
        delta_weights = [np.zeros(w.shape) for w in self.weights]
        delta_biases = [np.zeros(b.shape) for b in self.weights]
        delta = np.subtract(activated_z_values[-1], training_label)  # * get_sigmoid_prime(activated_z_values[-1])
        delta_biases[-1] = delta
        delta_weights[-1] = np.dot(delta, activated_z_values[-2].transpose())
        for layer in range(2, len(self.layers)):
            z = raw_z_values[- layer]
            delta = np.dot(self.weights[- layer + 1].transpose(), delta) * get_sigmoid_prime(z)
            delta_biases[- layer] = delta
            delta_weights[- layer] = np.dot(delta, activated_z_values[- layer - 1].transpose())
        return delta_weights, delta_biases

    def feed_forward(self, training_data: np.ndarray, evaluate: bool = False, dropout: List[np.ndarray] = None,
                     apply_dropout: bool = False
                     ) -> (List[np.ndarray], List[np.ndarray]):
        """
        x = transforming training data from (784,) to (1, 784) to (784, 1)
        we add first x to activated z values because we need it to back propagate the error of the first layer

        only neurons from hidden layer are dropped out, since tests have proven this give a faster good accuracy

        :param training_data: a single instance of training data, an array of 784 elements
        :param evaluate: if True, then functions was called to predict test instance
        :param dropout: dropout list
        :param apply_dropout: if True, then dropout should be applied
        :return: raw and activated z values, should be ([(100, 1), (10, 1)] , [(100, 1), (10, 1)] )
        """
        x = np.array([training_data]).transpose()
        raw_z_values = []
        activated_z_values = [x]
        for i in range(len(self.layers) - 1):
            weights = self.weights[i]
            z = np.dot(weights, x) + self.biases[i]

            if i != len(self.layers) - 2:
                x = apply_sigmoid(z)
                if apply_dropout:
                    z = np.multiply(dropout[i] * z, 1 / (1 - self.dropout_probability))
            else:
                x = soft_max(z)
            raw_z_values.append(z)
            activated_z_values.append(x)
        return raw_z_values, activated_z_values

    def evaluate(self, testing_data: np.ndarray, testing_labels: np.ndarray) -> float:
        """

        :param testing_data: np array of data
        :param testing_labels: np array of correct labels
        :return: #correctly classified/#total instances
        """
        correctly_classified = 0
        for i in range(testing_data.shape[0]):
            raw_values, activated_values = self.feed_forward(testing_data[i], evaluate=True)
            raw_values = raw_values[-1]
            activated_values = activated_values[-1]
            if np.argmax(raw_values) == np.argmax(testing_labels[i]):
                correctly_classified += 1
        return correctly_classified / testing_data.shape[0]

    def save_to_file(self):
        """
        saves neural network to file

        :return: nothing
        """
        dictionary = {
            "layers": self.layers,
            "weights": self.weights,
            "biases": self.biases
        }
        with open("neural_network.txt", "wb") as file:
            pickle.dump(dictionary, file, protocol=pickle.HIGHEST_PROTOCOL)
