#!/usr/bin/python3

import math
import urllib
import urllib.request
import string
import numpy as np


def print_hi(name):
    print(f'Hi, {name}')


def is_prime(x):
    """
    function that searches a number's divisor from 2 to int(sqrt(number))
    :param x: should be an int
    :return: True, if x is Prime, False otherwise
    """
    x = int(x)
    if x < 1:
        return False
    if x == 1:
        return False
    if x == 2:
        return True
    if x % 2 == 0:
        return False
    max_divisor = int(math.sqrt(x))
    divisor = 3
    while divisor <= max_divisor:
        if x % divisor == 0:
            return False
        divisor += 2
    else:
        return True


def read_file_from_url(target_url):
    """
    reads url from file and returns a string with the file
    :param target_url: valid url
    :return: string containing text from url
    """
    file = urllib.request.urlopen(target_url)
    text = ""
    for line in file:
        text += line.decode("utf-8")
    return text


def order_words_by_text(target_text):
    """
    splits string in words then iterates through resulted lists and strips punctuation from each word
    then sorts the string ignoring capital letters
    :param target_text: multi-words string
    :return: alphabetically ordered list of words
    """
    text_words = [word.strip(string.punctuation) for word in target_text.split()]
    text_words.sort(key=lambda v: v.upper())
    return text_words


def numpy_4():
    """
    Creati un vector de dimensiune 5 cu numere aleatoare:
        a. Afisati produsul scalar intre matricea definite la punctul 3 si vectorul curent
    """
    matrix_1 = np.random.rand(5, 5)
    vector_1 = np.random.rand(5)
    scalar_prod = np.dot(matrix_1, vector_1)
    print("a)")
    print("Matrix:\n", matrix_1)
    print("Vector:\n", vector_1)
    print("Scalar product = ", scalar_prod)


def numpy_3():
    """
    Creati o matrice cu numere aleatoare intre 0 si 1 de dimensiune 5x5.
        a. Afisati transpusa matricei
        b. Afisati inversa matricei si determinantul.
    """
    matrix_1 = np.random.rand(5, 5)
    transpose_1 = np.transpose(matrix_1)
    print("a)")
    print("Matrix:\n", matrix_1)
    print("Matrix transpose:\n", transpose_1)
    inverse_1 = np.linalg.inv(matrix_1)
    determinant_1 = np.linalg.det(matrix_1)
    print("b)")
    print("Matrix inverse:\n", inverse_1)
    print("Matrix determinant:\n", determinant_1)


def numpy_2():
    """
    Creati doi vectori cu numere aleatoare intre 0 si 1 de aceleasi dimensiuni.
        a. Afisati care dintre cei doi vectori are suma elementelor mai are.
        b. Adunati cei doi vectori. Inmultiti cei doi vector (vectorial si scalar).
        c. Calculati radical din fiecare element din vector
    """
    v_1 = np.random.rand(3)
    v_2 = np.random.rand(3)
    s1 = v_1.sum()
    s2 = v_2.sum()
    print("a)")
    print("Random vector 1:\n", v_1)
    print("Vector 1 sum = ", s1)
    print("Random vector 2:\n", v_2)
    print("Vector 2 sum = ", s2)
    if s1 > s2:
        print("Vector 1 is summier")
    elif s2 > s1:
        print("Vector 2 is summier")
    else:
        print("Vector are equal")
    vector_sum = np.add(v_1, v_2)
    cross_prod = np.cross(v_1, v_2)
    vector_prod = np.outer(v_1, v_2)
    scalar_prod = np.dot(v_1, v_2)
    print("b)")
    print("Vectors' sum = ", vector_sum)
    print("The cross product of 2 vectors = ", cross_prod)
    print("The outer product of 2 vectors = ", vector_prod)
    print("The scalar product of 2 vectors = ", scalar_prod)
    print("c)")
    for element in v_1:
        print("sqrt(" + str(element) + ") = ", math.sqrt(element))
    else:
        print("Done")
    for element in vector_sum:
        print("sqrt(" + str(element) + ") = ", math.sqrt(element))
    else:
        print("Done")


def numpy_1():
    """
    Creati matricea si vectorul de mai sus in numpy.
        a. Afisati doar ultimele 2 coloane din primele 2 randuri ale matricei
        b. Afisati ultimele 2 elemente din vector
    """
    matrix_1 = np.array([[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]], np.int32)
    matrix_2 = np.array([[2], [-5], [7], [-10]], np.int32)
    print("a)\n", matrix_1[0:2, 0:2])
    print("b)\n", matrix_2[-2:])
    return matrix_1, matrix_2


def exercise_3():
    """
    Data matricea si vectorul, afisati rezulatul produsului lor scalar.
    """
    matrix_1 = [
        [1, 2, 3, 4],
        [11, 12, 13, 14],
        [21, 22, 23, 24]
    ]
    matrix_2 = [
        [2],
        [-5],
        [7],
        [-10]
    ]
    matrix_3 = [
        [0],
        [0],
        [0]
    ]
    for i in range(len(matrix_1)):
        for j in range(len(matrix_2[0])):
            for k in range(len(matrix_2)):
                # print('i=', i, '\tj=', j, '\tk=', k)
                matrix_3[i][j] += matrix_1[i][k] * matrix_2[k][j]
    matrix_3 = np.array(matrix_3)
    print(matrix_3)


def exercise_2():
    """
    Ordonati cuvintele din fisier-ul urmator: "https://profs.info.uaic.ro/~rbenchea/rn/Latin-Lipsum.txt"
    """
    url = "https://profs.info.uaic.ro/~rbenchea/rn/Latin-Lipsum.txt"
    url_text = read_file_from_url(url)
    print(url_text)
    ordered_words = order_words_by_text(url_text)
    print(ordered_words)


def exercise_1():
    """
    Testati ca un numar este prim
    """
    number = input('Please enter a number:')
    number = int(number)
    print(number, ' is Prime = ', is_prime(number))


# exercise_1()
# exercise_2()
# exercise_3()
# numpy_1()
numpy_2()
# numpy_3()
# numpy_4()
if __name__ == '__main__':
    print_hi('PyCharm')
