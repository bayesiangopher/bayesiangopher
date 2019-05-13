import numpy as np
from numpy import linalg as LA
from common import *


@timer(cnt=count)
def create_random_matrix(shape: tuple) -> np.ndarray:
    """
    Создает матрицу с рандомными элементами [0;100)
    :param n: int - shape.
    :return:
    - <class 'numpy.ndarray'> - рандомная матрица.
    """

    return np.random.rand(shape[0]*shape[1]).reshape(shape[0], shape[1]) * 100


@timer(cnt=count)
def scale_matrix(a: np.ndarray, alpha: float):
    """
    Масштабирует переданную матрицу на alpha.
    :param a: np.ndarray - матрица для масштабирования;
    :param alpha: float - параметр масштабирования.
    """

    a * alpha


@timer(cnt=count)
def transpose_matrix(a: np.ndarray) -> np.ndarray:
    """
    Транспонирует переданную матрицу.
    :param a: np.ndarray - матрица для транспонирования.
    :return:
    -     - <class 'numpy.ndarray'> - транспонированная матрица a.
    """

    return a.transpose()


@timer(cnt=count)
def addition_of_matrices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Считает сумму двух матриц.
    :param a: np.ndarray - слагаемое;
    :param b: np.ndarray - слагаемое.
    :return:
    - np.ndarray - сумма.
    """

    return np.add(a, b)


@timer(cnt=count)
def subtract_of_matrices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Считает разницу двух векторов.
    :param a: np.ndarray - уменьшаемое;
    :param b: np.ndarray - вычитаемое.
    :return:
    - np.ndarray - разность.
    """

    return np.subtract(a, b)


@timer(cnt=count)
def dot_of_matrices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Считает произведение двух векторов.
    :param a: np.ndarray - множитель;
    :param b: np.ndarray - множитель.
    :return:
    - np.ndarray - произведение.
    """

    return np.dot(a, b)


@timer(cnt=count)
def determinant_of_matrix(a: np.ndarray) -> float:
    """
    Считает детерминант для переданной матрицы.
    :param a: np.ndarray - матрицы для поиска детерминанта.
    :return:
    - float - детерминант матрицы.
    """

    return LA.det(a)


@timer(cnt=count)
def eigens_of_matrix(a: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Находит собственные числа и правые собственные вектора.
    :param a: np.ndarray - матрицы для поиска собственных знач. и век..
    :return:
    - (np.ndarray, np.ndarray) - вектор собственных значений и
    матрица собственных векторов.
    """

    return LA.eig(a)


@timer(cnt=count)
def SVD_of_matrix(a: np.ndarray) -> (np.ndarray, np.diag, np.ndarray):
    """
    Находит элементы SVD разложения матрицы a.
    :param a: np.ndarray - матрицы для нахождения SVD разложения.
    :return:
    - (np.ndarray, np.diag, np.ndarray) - U, S, V'.
    """

    return LA.svd(a)


@timer(cnt=count)
def cholesky_of_matrix(a: np.ndarray) -> np.ndarray:
    """
    Находит разложение Холецкого для переданной эрмитовой
    (симметричной) положительно-определенной матрицы.
    :param a: np.ndarray - эрмитовая (симметричная)
    положительно-определенная матрица.
    :return:
    - np.ndarray - нижнетреугольная матрица L.
    """

    return LA.cholesky(a)
