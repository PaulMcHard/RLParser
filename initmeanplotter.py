import os
import numpy as np
import random
import pandas as pd
from parse import parser
import matplotlib.pyplot as plt
#import plaidml.keras
#plaidml.keras.install_backend()
#import keras

def chunk_list(list, n):
    for i in range(0, len(list), n):
        yield list[i : i + n]

class dataset():
    def __init__(self, data):
        self.ycom = data[:,0]
        self.yfbk = data[:,1]
        self.init_mean = np.mean(np.absolute(self.ycom-self.yfbk))
        self.num_steps = len(self.ycom)
        self.errors = np.zeros(self.num_steps)
        self.mean_errors = []

    def pad_size(self, max_steps):
        difference = max_steps - self.num_steps
        filler = np.zeros(difference)
        self.ycom = np.concatenate((self.ycom, filler))
        self.yfbk = np.concatenate((self.yfbk, filler))
        self.errors = np.concatenate((self.errors, filler))
        self.num_steps = len(self.ycom)

actions = [1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0, -.1, -.2, -.3, -.4, -.5, -.6, -.7, -.8, -.9, -1]
action_size = len(actions)
dirname = os.getcwd()
dirname+='/data/'
data_parser = parser()
files = []
for file in os.listdir(dirname):
    if file.endswith(".DAT"):
        files.append(file)

x_file_data = []
y_file_data = []
for file in files:
    data_parser.parse_data(dirname+file)
    temp_x = data_parser.get_x().values
    temp_y = data_parser.get_y().values
    x_file_data.append(temp_x)
    y_file_data.append(temp_y)

set_length = 500
x_data = []
y_data = []
for file in x_file_data:
    sets = list(chunk_list(file, set_length))
    for set in sets:
        temp = dataset(set)
        if temp.num_steps < set_length:
            temp.pad_size(set_length)
        x_data.append(temp)

for file in y_file_data:
    sets = list(chunk_list(file, set_length))
    for set in sets:
        temp = dataset(set)
        if temp.num_steps < set_length:
            temp.pad_size(set_length)
        y_data.append(temp)

x_init_means = []
for set in x_data:
    x_init_means.append(set.init_mean)

y_init_means = []
for set in y_data:
    y_init_means.append(set.init_mean)

f, (ax1, ax2) = plt.subplots(1,2, sharey = True)
ax1.hist(x_init_means)
ax2.hist(y_init_means)
plt.show()
