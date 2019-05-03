from vec import *
from common import *


def create_random_vector_set():
    """
    Принтует в stdout результаты теста по созданию векторов.
    """

    @out
    def test(shape: int):
        vec = create_random_vector(shape)
        print(f"Created random vector with size {shape}.\n")
        print(f"Vector size: {vec.shape} \n")

    for shape in dictVecSize.values():
        test(shape)


def scale_vector_set():
    """
    Принтует в stdout результаты теста масштабирования векторов.
    """

    @out
    def test(shape: int):
        alpha = 5.25
        vec = np.random.rand(shape) * 100
        check = vec[0]
        scaled_vec = scale_vector(vec, alpha)
        print(f"Scaled random vector with size {shape}.\n")
        print(f"Alpha after scale: {scaled_vec[0]/check} \n")

    for shape in dictVecSize.values():
        test(shape)


def frobenius_norm_of_vector_set():
    """
    Принтует в stdout результаты теста по определению
    нормы Фробениуса векторов.
    """

    @out
    def test(shape: int):
        vec = np.random.rand(shape) * 100
        fro = frobenius_norm_of_vector(vec)
        print(f"Calculated Frobenius norm of vector with size {shape}.\n")
        print(f"Frobenius norm: {fro} \n")

    for shape in dictVecSize.value():
        test(shape)


def addition_of_vectors_set():
    """
    Принтует в stdout результаты теста по определению
    суммы двух векторов.
    """

    @out
    def test(shape: int):
        vec = np.random.rand(shape) * 100
        vec_2 = np.random.rand(shape) * 100
        result = addition_of_vectors(vec, vec_2)
        print(f"Calculated addition of vectors with size {shape}.\n")
        print(f"Result: {result} \n")

    for shape in dictVecSize.values():
        test(shape)


def subtract_of_vectors_set():
    """
    Принтует в stdout результаты теста по определению
    разницы двух векторов.
    """

    @out
    def test(shape: int):
        vec = np.random.rand(shape) * 100
        vec_2 = np.random.rand(shape) * 100
        result = subtract_of_vectors(vec, vec_2)
        print(f"Calculated subtract of vectors with size {shape}.\n")
        print(f"Result: {result} \n")

    for shape in dictVecSize.values():
        test(shape)


def dot_of_vectors_set():
    """
    Принтует в stdout результаты теста по определению
    произведения двух векторов.
    """

    @out
    def test(shape: int):
        vec = np.random.rand(shape) * 100
        vec_2 = np.random.rand(shape) * 100
        result = dot_of_vectors(vec, vec_2)
        print(f"Calculated multiplication of vectors with size {shape}.\n")
        print(f"Result: {result} \n")

    for shape in dictVecSize.values():
        test(shape)


if __name__ == '__main__':
    # create_random_vector_set()
    scale_vector_set()
    # frobenius_norm_of_vector_set()
    # addition_of_vectors_set()
    # subtract_of_vectors_set()
    # dot_of_vectors_set()
    pass
