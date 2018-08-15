import os
import numpy as np
import random
from parse import parser


def chunk_list(list, n):
    for i in range(0, len(list), n):
        yield list[i: i + n]


class dataset():
    def __init__(self, data):
        self.xcom = data[:, 0]
        self.ycom = data[:, 1]
        self.xfbk = data[:, 2]
        self.yfbk = data[:, 3]
        self.init_mean = (np.mean(np.absolute(self.xcom-self.xfbk)), np.mean(np.absolute(self.ycom-self.yfbk)))
        self.num_steps = len(self.xcom)  # num_steps is constant for x and y
        self.errors = np.zeros((2, self.num_steps))
        self.mean_errors = []

    def pad_size(self, max_steps):
        difference = max_steps - self.num_steps
        filler = np.zeros(difference)
        self.num_steps = max_steps
        self.xcom = np.concatenate((self.xcom, filler))
        self.xfbk = np.concatenate((self.xfbk, filler))
        self.ycom = np.concatenate((self.ycom, filler))
        self.yfbk = np.concatenate((self.yfbk, filler))
        self.errors = np.zeros((2, self.num_steps))


def prepare(chosen_file, set_length):
    dirname = os.getcwd()
    dirname += '../data/'
    data_parser = parser()
    files = []
    for file in os.listdir(dirname):
        if file.endswith(".DAT"):
            files.append(file)

    chosen_test_set = chosen_file
    file_data_train = []
    file_data_test = []
    for i in range(len(files)):
        data_parser.parse_data(dirname+files[i])
        temp = data_parser.get_all_com_fbk().values
        if i == chosen_test_set:
            file_data_test.append(temp)
        else:
            file_data_train.append(temp)

    train_sets = []
    test_sets = []
    for file in file_data_train:
        sets = list(chunk_list(file, set_length))
        for set in sets:
            temp = dataset(set)
            if temp.num_steps < set_length:
                temp.pad_size(set_length)
            if temp.init_mean[0] <= 15 and temp.init_mean[1] <= 65:
                train_sets.append(temp)

    random.shuffle(train_sets)

    for file in file_data_test:
        sets = list(chunk_list(file, set_length))
        for set in sets:
            temp = dataset(set)
            if temp.num_steps < set_length:
                temp.pad_size(set_length)
            if temp.init_mean[0] <= 15 and temp.init_mean[1] <= 65:
                test_sets.append(temp)

    return  train_sets, test_sets
