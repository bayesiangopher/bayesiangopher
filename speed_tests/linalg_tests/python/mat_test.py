# Тестирование работы библиотеки numpy с матрицами
# For every size from: (16384, 65536, 262144, 1048576):
# 1 - create random matrices [0;100);
# 2 - scale matrix;
# 3 - transposing of matrix;
# 4 - addition of matrices;
# 5 - subtract of matrices;
# 6 - dot of matrices;
# 7 - determinant of matrix;
# 8 - eigens of matrix;
# 9 - SVD decomposition of matrix;
# 10 - Cholesky decomposition of matrix.

# Для использования:
# - pip3 install -r requirements.txt
# - раскомментить нужные тесты.

from mat import *
from common import *


def create_random_matrix_set():
    """
    Принтует в stdout результаты теста по созданию матрицы.
    """

    @out
    def test(shape: tuple):
        A = create_random_matrix(shape)
        print(f"Created random matrix with size {shape}.\n")
        print(f"Matrix size: {A.shape} \n")

    for shape in dictMatSize.values():
        test(shape)


def scale_matrix_set():
    """
    Принтует в stdout результаты теста по созданию масштабированию матрицы.
    """

    @out
    def test(shape: tuple):
        alpha = 5.25
        a = create_random_matrix(shape)
        check = a[0][0]
        scaled_a = scale_matrix(a, alpha)
        print(f"Scaled random matrix with size {shape}.\n")
        print(f"Alpha after scale: {scaled_a[0][0]/check} \n")

    for shape in dictMatSize.values():
        test(shape)


def transpose_matrix_set():
    """
    Принтует в stdout результаты теста по транспонированию матрицы.
    """

    @out
    def test(shape: tuple):
        a = create_random_matrix(shape)
        b = transpose_matrix(a)
        print(f"Transposed random matrix with size {shape}.\n")

    for shape in dictMatSize.values():
        test(shape)


def addition_of_matrice_set():
    """
    Принтует в stdout результаты теста по сложению матриц.
    """

    @out
    def test(shape: tuple):
        a = create_random_matrix(shape)
        b = create_random_matrix(shape)
        c = addition_of_matrices(a, b)
        print(f"Calculated addition of matrices with size {shape}.\n")
        print(f"Result: {c} \n")

    for shape in dictMatSize.values():
        test(shape)


def subtract_of_matrices_set():
    """
    Принтует в stdout результаты теста по вычитанию матриц.
    """

    @out
    def test(shape: tuple):
        a = create_random_matrix(shape)
        b = create_random_matrix(shape)
        c = subtract_of_matrices(a, b)
        print(f"Calculated subtract of matrices with size {shape}.\n")
        print(f"Result: {c} \n")

    for shape in dictMatSize.values():
        test(shape)


def dot_of_matrices_set():
    """
    Принтует в stdout результаты теста по перемножению матриц.
    """

    @out
    def test(shape: tuple):
        a = create_random_matrix(shape)
        b = create_random_matrix(shape)
        c = dot_of_matrices(a, b)
        print(f"Calculated multiplication of matrices with size {shape}.\n")
        print(f"Result: {c} \n")

    for shape in dictMatSize.values():
        test(shape)


def determinant_of_matrix_set():
    """
    Принтует в stdout результаты теста по нахождени
    детерминанта матрицы.
    """

    @out
    def test(shape: tuple):
        a = create_random_matrix(shape)
        d = determinant_of_matrix(a)
        print(f"Calculated determinant of matrix with size {shape}.\n")
        print(f"Result: {d} \n")

    for shape in dictMatSize.values():
        test(shape)


def eigens_of_matrix_set():
    """
    Принтует в stdout результаты теста по нахождени
    собственных чисел и векторов (ортонормированных) матрицы.
    """

    @out
    def test(shape: tuple):
        a = create_random_matrix(shape)
        w, v = eigens_of_matrix(a)
        print(f"Calculated eigens of matrix with size {shape}.\n")
        print(f"Results: values - {w}, vectors - {v} \n")

    for shape in dictMatSize.values():
        test(shape)


def SVD_of_matrix_set():
    """
    Принтует в stdout результаты теста по нахождени
    SVD разложения матрицы.
    """

    @out
    def test(shape: tuple):
        a = create_random_matrix(shape)
        u, s, vh = SVD_of_matrix(a)
        print(f"Calculated SVD of matrix with size {shape}.\n")
        print(f"Results: "
              f"U - {u}, "
              f"S - {s},"
              f"V* - {vh} \n")

    for shape in dictMatSize.values():
        test(shape)


def cholesky_of_matrix_set():
    """
    Принтует в stdout результаты теста по нахождени
    разложения Холецкого матрицы.
    """

    @out
    def test(shape: tuple):
        a = create_random_matrix(shape)
        asym = np.dot(a, a.transpose())
        trim = cholesky_of_matrix(asym)
        print(f"Calculated Cholesky decomposition of matrix with size {shape}.\n")
        print(f"Results: {trim} \n")

    for shape in dictMatSize.values():
        test(shape)


if __name__ == '__main__':
    # create_random_matrix_set()
    # scale_matrix_set()
    # transpose_matrix_set()
    # addition_of_matrice_set()
    # subtract_of_matrices_set()
    # dot_of_matrices_set()
    # determinant_of_matrix_set()
    # eigens_of_matrix_set()
    # SVD_of_matrix_set()
    # cholesky_of_matrix_set()
    pass
