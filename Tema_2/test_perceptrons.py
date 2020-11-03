from typing import List, Tuple
import numpy as np
from perceptron import Perceptron


def test_perceptrons(perceptrons: List[Perceptron], test_set: Tuple[np.ndarray, np.ndarray]):
    """

    :param perceptrons: the 10 perceptrons to be tested
    :param test_set:
    :return: nothing
    """
    print("testing")
    with open("all_perceptrons.txt", "w") as fd:
        for i in perceptrons:
            fd.write(i.to_string())
        # i.to_image()

    correct_guesses = 0
    test_results = np.zeros(10)
    for i in range(test_set[0].shape[0]):
        for j in range(10):
            test_results[j] = perceptrons[j].predict_raw_value(test_set[0][i])
        # if i % 100 == 0:
        #     print(test_results)
        #     print(test_set[1][i])
        #     print(np.amax(test_results))
        if test_results[test_set[1][i]] == np.amax(test_results):
            correct_guesses += 1
    return correct_guesses / test_set[0].shape[0]
