#Test of my parser working in a 'main' file

import tensorflow as tf
import numpy as np
import pandas as pd
from parser import parser

myParse = parser()
newdata = myParse.parse_data()
x = myParse.get_x()
matrix_x = x.as_matrix()
diff=np.absolute(matrix_x[:,0]-matrix_x[:,1])
print(diff)
mean = np.mean(diff)
print(mean)

dataVar_tensor = tf.constant(matrix_x, dtype=tf.float32)
