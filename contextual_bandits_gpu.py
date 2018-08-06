#Test of my parser working in a 'main' file

import os
import tensorflow as tf
import numpy as np
import random
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

random.shuffle(datasets)

actions = [1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0, -.1, -.2, -.3, -.4, -.5, -.6, -.7, -.8, -.9, -1]
a_size = len(actions)


# Function below gives our reward system.
def check_error(e_n, e_nplus):
    # Check error subtracted from previous error
    result = e_n - e_nplus
    if result < 0:
        # return a positive reward.
        return 100
    elif result == 0:
        #return a neutral reward.
        return 0
    else:
        #return a negative reward.
        return -1

class agent():
    def __init__(self, lr, s_size, a_size):
        #The next two lines establish the feed-forward part of the network. This does the actual choosing.
        self.weights = tf.Variable(tf.ones([a_size])) #Basically the QTable
        self.chosen_action = tf.argmax(self.weights,0) #Chooses action for exploitative

        #The next set of lines establish the training procedure. We feed the reward and chosen action into the Network
        #to compute the loss, and use it to update the network.
        #Once I figure out what I'm doing with eager/keras this all becomes much simpler
        self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
        self.responsible_weight = tf.slice(self.weights,self.action_holder,[1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update = self.optimizer.minimize(self.loss)

#Now move on to training the agent, for which we need to actually invoke a tensorflow session
total_episodes = 1000   #Set total number of episodes to train the agent on. IRL this will be training on as much data as I can give it.
total_reward = np.zeros(a_size)
#epsilon = 0.6 #Set the chance of taking a random action
epsilon = 1.0       #Exploration Rate
max_epsilon = 1.0   #Exploration Probability at start
min_epsilon = 0.01  #Minimum exploration Rate
decay_rate = 0.01   #Exponential decayrate for exploration prob

overall_init_mean = 0
sum = 0
for set in datasets:
    sum += set.init_mean
overall_init_mean = sum/len(datasets)


mean_errors = np.zeros((len(datasets), total_episodes))

#Now move on to establising the agent
tf.reset_default_graph()

#Initialise the agent.
the_agent = agent(lr = 0.001, s_size = set_length, a_size = a_size)

init = tf.global_variables_initializer()

    #Launch the tensorflow graph
with tf.device('/cpu:0'):

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(total_episodes):

            temp_xfbk = np.zeros(set_length)
            temp_err = np.zeros(set_length)

            for s in range(len(datasets)):
                for i in range(set_length-1):
                    #Choose either a random action or onefrom our network.
                    if np.random.rand(1) < epsilon:
                        action = np.random.randint(a_size)
                    else:
                        action = sess.run(the_agent.chosen_action)

                    temp_xfbk[i-1] = datasets[s].xfbk[i-1] + actions[action]*np.sign(datasets[s].xcom[i-1]-datasets[s].xfbk[i-1]) # add 1, 0 or -1 to demonstrate accel, const or decel
                    temp_err[i-1] = np.absolute(datasets[s].xcom[i-1]-temp_xfbk[i-1])
                    temp_err[i] = np.absolute(datasets[s].xcom[i]-datasets[s].xfbk[i])
                    reward = check_error(temp_err[i-1], temp_err[i])


                    #Update the network.
                    _,resp,ww = sess.run([the_agent.update,the_agent.responsible_weight,the_agent.weights], feed_dict={the_agent.reward_holder:[reward],the_agent.action_holder:[action]})
                    #print(str(action)+" : "+str(reward))

                    #Update running tally of scores
                    total_reward[action] += reward
                    '''if i % 1000 == 0:
                        print("Running reward: "+str(total_reward))
                        print("xcommand is: "+str(xcom[i])+" and xfeedback: "+str(xfbk[i]))'''

                datasets[s].errors = temp_err
                n_mean_step = np.mean(np.absolute(datasets[s].errors))
                datasets[s].mean_errors.append(n_mean_step)
                mean_errors[s, epoch] = n_mean_step
                #delta_err = datasets[s].init_mean - n_mean_step
                #print("Initial mean error for dataset {} was: {} In this episode, ({}), Reinforcement Learning has changed this by {} to: {} ".format(s,datasets[s].init_mean, epoch, delta_err, n_mean_step))

            overall_mean = np.mean(mean_errors[:, epoch])
            prev_mean = np.mean(mean_errors[:, epoch-1])
            init_delta = np.absolute(overall_init_mean - overall_mean)
            epoch_delta = np.absolute(overall_mean - prev_mean)
            print("Initial mean error was: {}, reduced in epoch {} to {}. A change of {} from previous, {} from init.".format(overall_init_mean, epoch, overall_mean, epoch_delta, init_delta))
            #Reduce epsilon (because we need less and less exploration)    n_mean_step = np.mean(np.absolute(datasets[s].errors))
            epsilon = min_epsilon +(max_epsilon - min_epsilon)*np.exp(-decay_rate*epoch)

end_mean = []
for i in range(0,99):
    end_mean.append(np.mean(mean_errors[:,i]))
plt.plot(end_mean)
plt.ylabel('mean error per iteration')
plt.xlabel('Number of iterations')
plt.show()
