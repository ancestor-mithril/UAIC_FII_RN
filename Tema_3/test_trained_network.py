import pickle
import numpy as np
from neural_network import Network


def check_trained_network(file_name: str, testing_set: (np.ndarray, np.ndarray)):
    """
    tests an already-trained nn saved to file

    :param file_name:
    :param testing_set:
    :return:
    """
    testing_data = testing_set[0]
    testing_labels = np.array([[[1] if x == i else [0] for i in range(10)] for x in testing_set[1]])
    with open(file_name, "rb") as file:
        x = pickle.load(file)

    return Network(layers=x["layers"], weights=x["weights"], biases=x["biases"]).evaluate(testing_data, testing_labels)
