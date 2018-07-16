import os
import numpy as np
import random
import tensorflow as tf
import pandas as pd
from parser import parser
import matplotlib.pyplot as plt

class dataset():
    def __init__(self, data, action_size):
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

actions = [1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0, -.1, -.2, -.3, -.4, -.5, -.6, -.7, -.8, -.9, -1]
action_size = len(actions)
dirname = os.getcwd()
dirname+='/data/'
data_parser = parser()
files = []
for file in os.listdir(dirname):
    if file.endswith(".DAT"):
        files.append(file)

datasets = []
files = sorted(files)
for file in files:
    data_parser.parse_data(dirname+file)
    temp = data_parser.get_x().values
    datasets.append(dataset(temp, action_size))

max_steps = 0
for set in datasets:
    if max_steps <= set.num_steps:
        max_steps = set.num_steps + 1

state_size = max_steps
qtable = np.zeros((state_size, action_size))

for set in datasets:
    set.pad_size(max_steps)

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

# SPECIFY HYPERPARAMETERS
total_episodes = 100         #Total total_episodes
learning_rate = 0.001          #Learning Rate
gamma = 0.95                 #Discounting Rate

#Exploration parameters
epsilon = 0.6       #Exploration Rate
#max_epsilon = 1.0   #Exploration Probability at start
#min_epsilon = 0.01  #Minimum exploration Rate
#decay_rate = 0.01   #Exponential decayrate for exploration prob

#List of rewards
rewards = []

mean_errors = np.zeros((len(datasets), total_episodes))

for episode in range(total_episodes):
    #Reset the environment
    step = 0
    done = False
    total_rewards = 0

    for set in range(len(datasets)):

        for step in range(max_steps-1):
            #Choose an action a in the current world state s, can be exploitation or exploration
            ## First we randomise a number
            exp_exp_tradeoff = random.uniform(0,1)

            ##If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[step, :])

            #Else doing a random choice --> exploration
            else:
                action = np.random.randint(action_size)

            #Take the action a and observe the outcome state s' and reward r
            datasets[set].xfbk[step] += actions[action]*np.sign(datasets[set].xcom[step]-datasets[set].xfbk[step]) #add 1, 0 or -1 to demonstrate accel, const or decel
            datasets[set].errors[step] = np.absolute(datasets[set].xcom[step]-datasets[set].xfbk[step])
            datasets[set].errors[step+1] = np.absolute(datasets[set].xcom[step+1]-datasets[set].xfbk[step+1])
            reward = check_error(datasets[set].errors[step], datasets[set].errors[step+1])


            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            qtable[step,action] = qtable[step, action] + learning_rate*(reward + gamma * np.max(qtable[step+1, :]) - qtable[step,action])

            total_rewards += reward

        n_mean_step = np.mean(np.absolute(datasets[set].errors))
        datasets[set].mean_errors.append(n_mean_step)
        mean_errors[set, episode] = n_mean_step
        delta_err = datasets[set].init_mean - n_mean_step
        print("Initial mean error for dataset {} was: {} In this episode, ({}), Q-Learning has changed this by {} to: {} ".format(set,datasets[set].init_mean,episode, delta_err, n_mean_step))

    episode += 1
    #Reduce epsilon (because we need less and less exploration)
    #epsilon = min_epsilon +(max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    rewards.append(total_rewards)
plt.plot(mean_errors)
plt.ylabel('mean error per iteration')
plt.xlabel('Number of iterations')
plt.show()
