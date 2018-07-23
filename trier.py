#Test of my parser working in a 'main' file

import os
import tensorflow as tf
import numpy as np
import pandas as pd
from parse import parser
import matplotlib.pyplot as plt

def chunk_list(list, n):
    for i in range(0, len(list), n):
        yield list[i : i + n]

class dataset():
    def __init__(self, data):
        self.xcom = data[:,0]
        self.xfbk = data[:,1]
        self.init_mean = np.mean(np.absolute(self.xcom-self.xfbk))
        self.num_steps = len(self.xcom)
        self.errors = np.zeros(self.num_steps)
        self.mean_errors = []

    def pad_size(self, max_steps):
        difference = max_steps - self.num_steps
        filler = np.zeros(difference)
        self.xcom = np.concatenate((self.xcom, filler))
        self.xfbk = np.concatenate((self.xfbk, filler))
        self.errors = np.concatenate((self.errors, filler))
        self.num_steps = len(self.xcom)

dirname = os.getcwd()
dirname+='/data/'
data_parser = parser()
files = []
for file in os.listdir(dirname):
    if file.endswith(".DAT"):
        files.append(file)

file_data = []
for file in files:
    data_parser.parse_data(dirname+file)
    temp = data_parser.get_x().values
    file_data.append(temp)

set_length = 500
datasets = []
for file in file_data:
    sets = list(chunk_list(file, set_length))
    for set in sets:
        temp = dataset(set)
        if temp.num_steps < set_length:
            temp.pad_size(set_length)
        datasets.append(temp)
