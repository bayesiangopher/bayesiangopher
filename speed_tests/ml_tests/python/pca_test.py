import numpy as np
from sklearn.decomposition import PCA
from common import *


class PCATester:
    X = None,
    pca = None

    def __init__(self):
        pass

    def set_data(self, path: str):
        self.X = np.genfromtxt(path, delimiter=',')[1:]
        print(self.X.shape)

    def create_pca(self, n_components):
        self.pca = PCA(n_components=n_components)
        self.do_decomposition()

    @timer(cnt=count)
    def do_decomposition(self, n_components=2):
        """
        Декомпозирует данные (снижает размерность).
        """

        assert self.X is not None, "данные не найдены"
        self.X_decomposed = self.pca.fit_transform(self.X)

    def print_results(self):
        print(self.pca.X_decomposed)

    def get_coefs(self):
        assert self.pca, "модель не обучена"
        print(f"Mean vector: \n{self.pca.mean_}")
        print(f"Projection: \n{self.pca.components_}")
        print(f"Explained variance: \n{self.pca.explained_variance_}")
        print(f"Explained variance ratio: \n{self.pca.explained_variance_ratio_}")
        print(f"Singular values: \n{self.pca.singular_values_}")


def pca_test():
    """
    Принтует в stdout результаты теста декомпозиции pca.
    """

    @out
    def test(pca):
        pca.create_pca(2)
        pca.get_coefs()

    pca = PCATester()
    pca.set_data("../../../datasets/the_boston_housing_dataset.csv")
    test(pca)


if __name__ == "__main__":
    pca_test()