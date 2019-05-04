#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
import csv
import argparse
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pprint import pprint


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", nargs='?', default=[f"{os.getcwd()}"])
    return parser


class TFig(Figure):

    cs = dict(
        golang='b',
        python='g',
        c='k'
    )

    def __init__(self, *args, fig_title, **kwargs):
        super().__init__(*args, **kwargs)
        self.suptitle(fig_title, fontsize=14, ha='center')
        self.autofmt_xdate()

    @property
    def color_settings(self):
        return self.cs


class Plotter:

    source = None
    data_files = []
    datasets = []
    cases = dict()

    def __init__(self, source: str):
        """
        Создает списки со строками из файлов .csv, находящихся
        в каталоге по переданному source и пишет их в self.data_sets.
        Формат так грузить оперативку одним экземпляром класса -
        не самый правильный, но данных не так много, а поддержка
        ООП паттерна питона - наше все :-).
        :param source: str - путь к файлам с данными тестирования.
        """

        self.source = source
        df = os.listdir(path=source)
        csv.register_dialect("common_csv", delimiter=',', quoting=csv.QUOTE_NONE)
        for f in df:
            if ".csv" in f:
                self.data_files.append(f)
        for i, data_set in enumerate(self.data_files):
            with open(data_set, newline='\n') as ds:
                reader = csv.reader(ds, 'common_csv')
                self.datasets.append([])
                try:
                    for row in reader:
                        self.datasets[i].append(row)
                except csv.Error as e:
                    sys.exit(f"file {ds}, line {reader.line_num}: {e}")

    @property
    def datasets_count(self):
        return len(self.datasets)

    def datasets_printer(self, offset: int, *args):
        print(f"Общее кол-во датасетов: {self.datasets_count}.")
        limit = args[0] if args else self.datasets_count
        for ds in self.datasets[offset:limit + 1]:
            pprint(ds)

    def prepare_data(self):

        # Да, я сам знаю, что за такое "вон из профессии", но
        # нет времени думать, как это реализовать нормально.
        def form_data(self, case):
            if not (self.cases.get(time_pattern).get(case[0])):
                self.cases[time_pattern][case[0]] = [[case[3]], [case[4]]]
                self.cases[mem_pattern][case[0]] = [[case[3]], [case[5]]]
            else:
                self.cases[time_pattern][case[0]][0].append(case[3])
                self.cases[time_pattern][case[0]][1].append(case[4])
                self.cases[mem_pattern][case[0]][0].append(case[3])
                self.cases[mem_pattern][case[0]][1].append(case[5])

        for ds in self.datasets:
            for case in ds[1:]:
                time_pattern = f"{case[1]}-{case[2]}-time"
                mem_pattern = f"{case[1]}-{case[2]}-mem"
                if not(self.cases.get(time_pattern)):
                    self.cases[time_pattern] = dict()
                    self.cases[mem_pattern] = dict()
                    form_data(self, case)
                else:
                    form_data(self, case)

    def create_figs(self):
        self.prepare_data()
        for key, value in self.cases.items():
            fig = plt.figure(FigureClass=TFig, fig_title=key)
            ax = fig.subplots()
            ax.set_xlabel("Size (for vectors: N, for matrices: N*M)")
            ax.set_ylabel('Memory usage') if "mem" in key else ax.set_ylabel('Time')
            for k, v in value.items():
                color = TFig.cs.get(k)
                ax.plot(v[0], v[1], f"{color}o-", label=f"{k}")
            fig.legend(bbox_to_anchor=(0.3, 0.85), borderaxespad=0.)
            plt.savefig(f"{self.source}/{key}.png")
