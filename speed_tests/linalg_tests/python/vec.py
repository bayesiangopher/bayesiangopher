# Testing speed of operations
# For every size from: (1024, 16384, 65536, 131072, 262144, 524288):
# - create random vectors [0;100);
# - scale vector;
# - Frobenius norm of vector;
# - addition of vectors;
# - subtract of vectors;
# - dot of vectors.


import numpy as np
from numpy import linalg as LA
from common import *


@timer(count=count)
def create_random_vector(n: int) -> np.ndarray:
    """
    Создает вектор с рандомными элементами [0;100)
    :param n: int - shape.
    :return:
    - <class 'numpy.ndarray'> - рандомный вектор.
    """

    return np.random.rand(n) * 100


@timer(count=count)
def scale_vector(vec: np.ndarray, alpha: float) -> np.ndarray:
    """
    Масштабирует переданный вектор на alpha.
    :param vec: np.ndarray - вектор для масштабирования;
    :param alpha: float - параметр масштабирования.
    """

    return vec * alpha


@timer(count=count)
def frobenius_norm_of_vector(vec: np.ndarray) -> float:
    """
    Считает норму Фробениуса для переданного вектора.
    :param vec: np.ndarray - вектор для поиска нормы;
    :return:
    - float - норма Фробениуса переданного вектора.
    """

    return LA.norm(vec)


@timer(count=count)
def addition_of_vectors(vec: np.ndarray, vec_2: np.ndarray) -> np.ndarray:
    """
    Считает сумму двух векторов.
    :param vec: np.ndarray - слагаемое;
    :param vec_2: np.ndarray - слагаемое.
    :return:
    - np.ndarray - сумма.
    """

    return np.add(vec, vec_2)


@timer(count=count)
def subtract_of_vectors(vec: np.ndarray, vec_2: np.ndarray) -> np.ndarray:
    """
    Считает разницу двух векторов.
    :param vec: np.ndarray - уменьшаемое;
    :param vec_2: np.ndarray - вычитаемое.
    :return:
    - np.ndarray - разность.
    """

    return np.subtract(vec, vec_2)


@timer(count=count)
def dot_of_vectors(vec: np.ndarray, vec_2: np.ndarray) -> np.ndarray:
    """
    Считает произведение двух векторов.
    :param vec: np.ndarray - множитель;
    :param vec_2: np.ndarray - множитель.
    :return:
    - np.ndarray - произведение.
    """

    return np.dot(vec, vec_2)