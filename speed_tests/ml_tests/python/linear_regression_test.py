import numpy as np
from sklearn.linear_model import LinearRegression
from common import *


class LinearRegressionTester:

    X = None,
    y = None,
    X_test = None,
    y_test = None,
    lr = None

    def __init__(self):
        pass

    def set_data(self, path: str, path_test: str):
        train = np.genfromtxt(path, delimiter=',')[1:]
        train_test = np.genfromtxt(path_test, delimiter=',')[1:]
        self.X = train[:, 1:]
        self.y = train[:, 0]
        self.X_test = train_test[:, 1:]
        self.y_test = train_test[:, 0]

    def create_linear_regression(self):
        self.lr = LinearRegression()
        self.do_linear_regression()

    @timer(cnt=count)
    def do_linear_regression(self):
        """
        Восстанавливает линейную регрессию по данным из
        self.X, self.y. Сохраняет результаты в self.lr.
        """

        assert (self.X is not None and self.y is not None), "данные не найдены"
        self.lr.fit(self.X, self.y)

    def get_r_squared(self):
        assert self.lr, "модель не обучена"
        assert (self.X_test is not None and self.y_test is not None), "тестовые данные не переданы"
        return self.lr.score(self.X_test, self.y_test)


def linear_regression_test():
    """
    Принтует в stdout результаты теста восстановления регрессии.
    """

    @out
    def test(LR):
        LR.create_linear_regression()
        print("Регрессия восстановлена.")
        print(f"R^squared: {LR.get_r_squared()}.\n")

    LR = LinearRegressionTester()
    LR.set_data("../../../datasets/the_WWT_weather_10k_dataset.csv",
                "../../../datasets/the_WWT_weather_test_train.csv")
    test(LR)


if __name__ == "__main__":
    linear_regression_test()

