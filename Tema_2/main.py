import gzip
import pickle
from perceptron import Perceptron
import train_perceptrons as tp

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin')
f.close()


perceptrons = [Perceptron(value=i) for i in range(10)]
tp.train_perceptrons_together(perceptrons, train_set, test_set, niu=0.001, iterations=10)


# if __name__ == "__main__":
#     tp.train_perceptrons_subprocess(train_set, valid_set, iterations=10, niu=0.001)


# m1, m2 = eval(open("input.txt", 'r').read())
# print(m1, m2)
# for i in m2:
#     # print(i, i.index(1))
#     print(Perceptron(one_hot_vector=i).number)
