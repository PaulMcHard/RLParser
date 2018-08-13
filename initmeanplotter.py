import os
import numpy as np
import random
import pandas as pd
from parse import parser
import matplotlib.pyplot as plt


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


actions = [1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0, -.1, -.2, -.3, -.4, -.5, -.6, -.7, -.8, -.9, -1]
action_size = len(actions)
dirname = os.getcwd()
dirname += '/data/'
data_parser = parser()
files = []
for file in os.listdir(dirname):
    if file.endswith(".DAT"):
        files.append(file)

file_data_train = []
for i in range(len(files)):
    data_parser.parse_data(dirname+files[i])
    temp = data_parser.get_all_com_fbk().values
    file_data_train.append(temp)

set_length = 200
train_sets = []
for file in file_data_train:
    sets = list(chunk_list(file, set_length))
    for set in sets:
        temp = dataset(set)
        if temp.num_steps < set_length:
            temp.pad_size(set_length)
        if temp.init_mean[0] <= 15 and temp.init_mean[1] <= 65:
            train_sets.append(temp)

x_init_means = []
y_init_means = []
for set in train_sets:
    x_init_means.append(set.init_mean[0])
    y_init_means.append(set.init_mean[1])

print("There are {} datasets in the value range, with a mean of [ {} , {} ]".format(len(train_sets), np.mean(x_init_means), np.mean(y_init_means)))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.hist(x_init_means)
ax2.hist(y_init_means)
plt.show()
