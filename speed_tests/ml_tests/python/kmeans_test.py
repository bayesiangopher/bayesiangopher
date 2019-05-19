import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from common import *


class KMeansTester:

    train = None,
    train_test = None,
    kmeans = None

    def __init__(self):
        pass

    def set_data(self, path: str, path_test: str):
        self.train = np.genfromtxt(path, delimiter=',')[1:]
        self.train_test = np.genfromtxt(path_test, delimiter=',')[1:]
        print(self.train.shape)

    def create_kmeans(self):
        self.kmeans = KMeans(n_clusters=3, random_state=0)
        self.do_kmeans()

    @timer(cnt=count)
    def do_kmeans(self):
        """
        Обучая кластеризатор kmeans.
        """

        assert self.train is not None, "данные не найдены"
        self.kmeans.fit(self.train)

    def get_score(self):
        assert self.kmeans, "модель не обучена"
        assert self.train_test is not None, "тестовые данные не переданы"
        return self.kmeans.score(self.train_test)

    def get_pictures(self):
        unique_labels = set(self.kmeans.labels_)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.use_sticky_edges = False
        ax.margins(0.07)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_title(f"clustered_data_set")
        for c, d in zip(self.kmeans.labels_, self.train):
            plt.scatter(d[0], d[1], color=colors[c])
        plt.savefig(f"kmeans_clusters_test.png")


def kmeans_test():
    """
    Принтует в stdout результаты теста обучения kmeans классификатора.
    """

    @out
    def test(KMEANS):
        KMEANS.create_kmeans()
        print(f"Score: {KMEANS.get_score()}.\n")

    KMEANS = KMeansTester()
    KMEANS.set_data("../../../datasets/the_xclara_cluster_2.5k_dataset.csv",
                    "../../../datasets/the_xclara_cluster_test_train.csv")
    test(KMEANS)
    KMEANS.get_pictures()


if __name__ == "__main__":
    kmeans_test()
