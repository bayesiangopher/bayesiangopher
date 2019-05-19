import numpy as np
from sklearn.linear_model import LogisticRegression
from common import *


class LogisticRegressionTester:

    X = None,
    y = None,
    X_test = None,
    y_test = None,
    logr = None

    def __init__(self):
        pass

    def set_data(self, path: str, path_test: str):
        train = np.genfromtxt(path, delimiter=',')[1:]
        train_test = np.genfromtxt(path_test, delimiter=',')[1:]
        self.X = train[:, 1:]
        self.y = train[:, 0]
        self.X_test = train_test[:, 1:]
        self.y_test = train_test[:, 0]
        print(self.X.shape)

    def create_linear_regression(self):
        self.logr = LogisticRegression(random_state=0, solver='lbfgs')
        self.do_logistic_regression()

    @timer(cnt=count)
    def do_logistic_regression(self):
        """
        Обучает логистический классификатор.
        """

        assert (self.X is not None and self.y is not None), "данные не найдены"
        self.logr.fit(self.X, self.y)

    def get_score(self):
        assert self.logr, "модель не обучена"
        assert (self.X_test is not None and self.y_test is not None), "тестовые данные не переданы"
        return self.logr.score(self.X_test, self.y_test)


def logistic_regression_test():
    """
    Принтует в stdout результаты теста обучения логистического классификатора.
    """

    @out
    def test(LOGR):
        LOGR.create_linear_regression()
        print("Регрессия восстановлена.")
        print(f"R^squared: {LOGR.get_score()}.\n")

    LOGR = LogisticRegressionTester()
    LOGR.set_data("../../../datasets/the_breast_canser_500rows_dataset.csv",
                "../../../datasets/the_breast_canser_test_train.csv")
    test(LOGR)


if __name__ == "__main__":
    logistic_regression_test()
