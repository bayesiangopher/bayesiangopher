# Определение декораторов
# и данных для тестирования

import os
import time
from memory_profiler import profile

DEFAULT = 10
dictVecSize = {"Size=1024": 1024,
               "Size=16384": 16384,
               "Size=65536": 65536,
               "Size=131072": 131072,
               "Size=262144": 262144,
               "Size=524288": 524288}
dictMatSize = {"Size=1024": (32, 32),
               "Size=16384": (128, 128),
               "Size=65536": (256, 256),
               "Size=262144": (512, 512),
               "Size=1048576": (1024, 1024)}

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
            for n in range(cnt):
                value = func(*args, **kwargs)
            end = time.time()
            runtime = (end - start)/cnt
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
