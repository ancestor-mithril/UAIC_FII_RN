import random
import numpy as np
from neural_network import Network
from data_loader import wrap_data, load_data
from test_trained_network import check_trained_network
from utils import unison_shuffled_copies, to_image, blur_image, generate_more_training_data

train_set, valid_set, test_set = load_data()

# to_image(train_set[0][0].reshape([28, 28]))  # testing data augmentation: blurring images


# train_set = generate_more_training_data(train_set)

# Network([784, 100, 10]).train_neural_network(train_set, iterations=10, testing_set=test_set, apply_dropout=True,
#                                              dropout_rate=0.2)

print(check_trained_network("neural_network_trained_on_blurred_data_with_dropout_rate_20_10_gen.txt", test_set))
