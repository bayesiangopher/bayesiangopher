# Определение декораторов
# и данных для тестирования

import os
import time
from memory_profiler import profile

DEFAULT = 10

try:
    count = os.environ["PYTHON_TESTS_COUNT"]
except KeyError:
    count = DEFAULT


def timer(cnt):
    """
    Таймер для замера скорости работы тестируемых функций.
    :param cnt: int - кол-во прогонов функции.
    """

    def wrp(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            print(f"Start: {start}")
            for n in range(cnt):
                value = func(*args, **kwargs)
            end = time.time()
            print(f"End: {end}")
            runtime = (10**9)*(end - start)/cnt
            print(f"\nTime of work {func.__name__}({args}): {runtime}\n")
            return value
        # декоратор profile - изменяет расход памяти исполняемым кодом.
        return profile(precision=10)(wrapper)
    return wrp


def out(func):
    """
    Оборачивает функцию-тест в тестовые записи.
    """

    def wrapper(*args):
        print("BEGIN OF TEST")
        func(*args)
        print("END OF TEST\n")
    return wrapper
