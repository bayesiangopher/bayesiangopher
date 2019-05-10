# Tiny handmade PCA (principal component analysis)

# Немного в формате ликбеза-объяснения:
# Метод основан на трактовке дисперсии как характеристики
# формы распределения случайной величины. Для векторов случайных величин,
# образованных наблюдаемыми фичами, находится матрица ковариации,
# характеризующая форму случайных векторов. По матрице ковариаций можно
# понять форму векторов случайных величин (по диагонали находятся
# дисперсии каждой случайной величины, характеризующие
# размеры объема векторов по каждой из осей). Далее необходимо
# определить вектор, при котором макисимизировался бы размер проекции
# наших случайных векторпов на него. Такой вектор - собственный вектор
# имеющий макисмальное собственное число (из отношения Рэлея и формулы
# проекции вектора на вектор).

import numpy as np

# Вообще не нужен тут класс, но захотелось, вот функция:


def pca_func(data: np.ndarray, components: int) -> np.ndarray:
    centred_data = data
    for idx, random_var in enumerate(data):
        random_var = random_var - random_var.mean()
        centred_data[idx] = random_var
    n, m = centred_data.shape
    print(f"Centred data:\n {centred_data}")
    assert np.allclose(centred_data.mean(axis=1), np.zeros(n))
    covariance_matrix = np.cov(centred_data)
    print(f"Covariance matrix:\n {covariance_matrix}\n\n")
    eigen_vals, eigen_vecs = np.linalg.eig(covariance_matrix)
    print(f"Eigne vectors:\n {eigen_vecs}\n\n")
    # components_vecs = np.array([[eigen_vecs.transpose()[0]], [np.linalg.norm(eigen_vecs[0])]])
    # for vec in eigen_vecs.transpose():
    #     norm = np.linalg.norm(vec)
    #     if components_vecs.shape[0] == components and norm > np.amax(components_vecs[1]):
    #         idx = np.where(components_vecs[1] == np.amin(components_vecs[1]))
    #         components_vecs[1][idx] = norm
    #         components_vecs[0][idx] = vec
    #     elif components_vecs.shape[0] < components:
    #         np.append(components_vecs[0], vec, axis=0)
    #         np.append(components_vecs[1], norm, axis=1)
    print(f"Eigen vec:\n {eigen_vecs[:, 1]}\n\n")
    X_pca = np.dot(eigen_vecs[:, 1], centred_data)
    return X_pca


def svd(X):
    U, Sigma, Vh = np.linalg.svd(X,
                                 full_matrices=False, # It's not necessary to compute the full matrix of U or V
                                 compute_uv=True)
    # Transform X with SVD components
    print(U)
    print(np.diag(Sigma))
    X_svd = np.dot(U, np.diag(Sigma))
    return X_svd


class PCA:

    def __init__(self):
        pass

    @staticmethod
    def create_date() -> np.ndarray:
        # first feature:
        x = np.arange(1, 11)
        # second feature:
        y = 2 * x + np.random.randn(10) * 2
        # data set:
        dataset = np.vstack((x, y))
        # print generated data:
        print(f"Dataset:\n {dataset}\n\n")
        return dataset

    @staticmethod
    def moments(data: np.ndarray) -> tuple:
        """
        :return:
        - (centred_data, means vector) - centred_data and means.
        Mean we can understand as the center of gravity.
        """

        centred_data = (data[0] - data[0].mean(), data[1] - data[1].mean())
        m = (data[0].mean(), data[1].mean())
        print(f"Centred data:\n {centred_data}")
        print(f"Mean vector:\n {m}\n\n")
        return centred_data, m

    @staticmethod
    def covariance_matrix(data: np.ndarray) -> np.ndarray:
        """
        :return:
        - cov_mat: np.array - covariance matrix of X and Y
        random values.
        """

        cov_mat = np.cov(data)
        print(f"Covariance matrix:\n {cov_mat}")
        print(f"Variance of X:\n {cov_mat[0, 0]}")
        print(f"Variance of Y:\n {cov_mat[1, 1]}")
        print(f"Covariance X and Y:\n {cov_mat[0, 1]}\n\n")
        return cov_mat

    @staticmethod
    def eigens(data: np.ndarray) -> tuple:
        eigenval, eigenvec = np.linalg.eig(data)
        print(f"Eigen values:\n {eigenval}")
        print(f"Eigen vectors:\n {eigenvec}\n\n")
        return eigenval, eigenvec

    def projection(self, data, centred_data):
        _, eigenvecs = self.eigens(data)
        v = eigenvecs[:, 1]
        print(f"Eigen vector:\n {v}\n\n")
        Xnew = np.dot(v, centred_data)
        print(f"New data:\n {Xnew}\n\n")
        return Xnew

    @staticmethod
    def check(X, Xnew, m, eigenvec):
        Xrestored = np.dot(Xnew[5], eigenvec) + m
        print(f"Restored:\n {Xrestored}")
        print(f"Original:\n {X[:, 5]}")


if __name__ == "__main__":
    PCA_tiny = PCA()
    data = PCA_tiny.create_date()
    # data_1 = np.array([
    #     1., 2., 3., 4., 5., 6., 7., 8., 9., 10.
    # ]),
    # data_2 = np.array([
    #     2.73446908, 4.35122722, 7.21132988,
    #     11.24872601, 9.58103444, 12.09865079,
    #     13.78706794, 13.85301221, 15.29003911, 18.0998018
    # ])
    # data = np.vstack((data_1, data_2))
    # centred_data, mean_matrix = PCA_tiny.moments(data)
    # cov_mat = PCA_tiny.covariance_matrix(data)
    # eigenvals, eigenvecs = PCA_tiny.eigens(cov_mat)
    # Xnew = PCA_tiny.projection(cov_mat, centred_data)
    # PCA_tiny.check(data, Xnew, mean_matrix, eigenvecs[:, 1])
    #
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=1)
    # Xnew_sklearn = pca.fit_transform(np.transpose(data))
    #
    # print(f"Our result:\n {Xnew}")
    # print(f"Sklearn result:\n {Xnew_sklearn}")

    Xnew = pca_func(data, 1)
    print(f"Func result:\n {Xnew}")

    Xnew = pca_func(data, 1)
    print(f"Func result:\n {Xnew}")

