import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib
from common import *


class DataGaner():

    def __init__(self, type="complex", size=250):
        self.size = size
        self.type = type
        self.create_data()

    def create_data(self):
        if self.type == "complex":
            self.data = [(np.random.randn()/6 - 1, np.random.randn()/6 - 1) for i in range(self.size)]
            self.data.extend([(np.random.randn()/4 + 2.5, np.random.randn()/5) for i in range(self.size)])
            self.data.extend([(np.random.randn()/3 - 2, np.random.randn()/10 + 1) for i in range(self.size)])
            self.data.extend([(np.random.randn()/50 - 2, np.random.randn() + 1) for i in range(self.size)])
            self.data.extend([(np.random.randn()/5 + 1, np.random.randn()/2 + 1) for i in range(self.size)])
            self.data.extend([(i/25 - 1, + np.random.randn()/20 - 3) for i in range(self.size)])
            self.data.extend([(i/25 - 2.5, 9 - (i/50 - 2)**2 + np.random.randn()/20) for i in range(self.size)])
            self.data.extend([(i/25 - 2.5, 6 + (i/50 - 2)**2 + np.random.randn()/2) for i in range(self.size)])
            self.data = np.array(self.data)
        else:
            self.data = [(np.random.randn()/6, np.random.randn()/6) for i in range(150)]
            self.data.extend([(np.random.randn()/4 + 2.5, np.random.randn()/5) for i in range(150)])
            self.data.extend([(np.random.randn()/5 + 1, np.random.randn()/2 + 1) for i in range(150)])
            self.data.extend([(i/25 - 1, + np.random.randn()/20 - 1) for i in range(100)])
            self.data.extend([(i/25 - 2.5, 3 - (i/50 - 2)**2 + np.random.randn()/20) for i in range(150)])
            self.data = np.array(self.data)

    def print_data(self):
        print(self.data[:])

    def plot_data(self):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.use_sticky_edges = False
        ax.margins(0.07)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.scatter(self.data[:, 0], self.data[:, 1])
        ax.set_title(f"generated_data_set")
        plt.savefig(f"dbscan_data_test_simple.png")


class DBSCANTester(DataGaner):

    dbscan = None
    data = None

    def __init__(self, type="complex", size=250):
        super().__init__(type, size)
        self.plot_data()

    def create_clusters(self, eps=0.3, min_samples=7):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.do_clustering()
        unique_labels = set(self.dbscan.labels_)
        print(f"Кол-во уникальных меток: {len(unique_labels)}")
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.use_sticky_edges = False
        ax.margins(0.07)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_title(f"clustered_data_set")
        for c, d in zip(self.dbscan.labels_, self.data):
            plt.scatter(d[0], d[1], color=colors[c])
        plt.savefig(f"dbscan_clusters_test_simple.png")

    @timer(cnt=count)
    def do_clustering(self):
        """
        Маркирует кластеры алгоритмом DBCSAN.
        """

        assert self.data is not None, "данные не найдены"
        self.dbscan.fit(self.data)


def dbscan_test():
    """
    Принтует в stdout результаты теста кластеризации
    алгоритмом DBSCAN.
    """

    @out
    def test(DBSCAN):
        DBSCAN.create_clusters()
        print("Кластеризация выполнена.")

    DBSCAN = DBSCANTester(size=500)
    print(f"Размер обучающей выборки: {DBSCAN.data.shape}")
    test(DBSCAN)


if __name__ == "__main__":
    dbscan_test()
