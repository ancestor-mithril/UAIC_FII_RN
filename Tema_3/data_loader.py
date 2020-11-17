import gzip
import pickle
import numpy as np


def load_data():
    """
    loads data from mnist dataset

    :return: (np.array[50000][784], np.array[50000]) * 3
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin')
    f.close()
    return train_set, valid_set, test_set


def wrap_data():
    train_set, valid_set, test_set = load_data()
    # TODO: check if useless and remove
