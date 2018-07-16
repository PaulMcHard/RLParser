#Test of my parser working in a 'main' file
from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import numpy as np
from parser import parser

tf.enable_eager_execution()

print("Tensorflow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

#Function below gives our reward system.
def check_error(e_n, e_nplus):
    #Check error subtracted from previous error
    result = e_n - e_nplus
    if result < 0:
        #return a positive reward.
        return 100
    elif result == 0:
        #return a neutral reward.
        return 0
    else:
        #return a negative reward.
        return -1

dirname = os.getcwd()
data_parser = parser()
data_parser.parse_data(dirname+'/data/9cadTEST3.DAT')
x = data_parser.get_x().values
xcom = x[:,0]
xfbk = x[:,1]
o_mean = np.mean(np.absolute(xcom-xfbk))
prog_means = []
num_steps = len(xcom)
actions = [1,0.5,0.1,0,-0.1,-0.5, -1]
a_size = len(actions)
errors = np.zeros(num_steps)
