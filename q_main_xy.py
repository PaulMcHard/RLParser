import os
import numpy as np
import random
import pandas as pd
from parse import parser
import matplotlib.pyplot as plt
#import keras

def chunk_list(list, n):
    for i in range(0, len(list), n):
        yield list[i : i + n]

class dataset():
    def __init__(self, data):
        self.xcom = data[:,0]
        self.ycom = data[:,1]
        self.xfbk = data[:,2]
        self.yfbk = data[:,3]
        self.init_mean = (np.mean(np.absolute(self.xcom-self.xfbk)),np.mean(np.absolute(self.ycom-self.yfbk)))
        self.num_steps = len(self.xcom) #num_steps is constant for x and y
        self.errors = np.zeros((2,self.num_steps))
        self.mean_errors = []

    def pad_size(self, max_steps):
        difference = max_steps - self.num_steps
        filler = np.zeros(difference)
        self.num_steps = max_steps
        self.xcom = np.concatenate((self.xcom, filler))
        self.xfbk = np.concatenate((self.xfbk, filler))
        self.ycom = np.concatenate((self.ycom, filler))
        self.yfbk = np.concatenate((self.yfbk, filler))
        self.errors = np.zeros((2,self.num_steps))

actions = [1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0, -.1, -.2, -.3, -.4, -.5, -.6, -.7, -.8, -.9, -1]
action_size = len(actions)
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
    temp = data_parser.get_all_com_fbk().values
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

state_size = set_length
qtable = np.zeros((2,state_size, action_size))
#x_qtable = np.zeros((state_size, action_size))
#y_qtable = np.zeros((state_size, action_size))

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
total_epochs = 500         #Total total_epochs
learning_rate = 0.003          #Learning Rate
gamma = 0.95                 #Discounting Rate

#Exploration parameters
epsilon = 1.0       #Exploration Rate
max_epsilon = 1.0   #Exploration Probability at start
min_epsilon = 0.01  #Minimum exploration Rate
decay_rate = 0.01   #Exponential decayrate for exploration prob

#List of rewards
rewards = []

overall_init_mean = np.zeros(2)
sum = np.zeros(2)
for set in datasets:
    for i in range(0,1):
        sum += set.init_mean
overall_init_mean = sum/len(datasets)

x_mean_errors = np.zeros((len(datasets), total_epochs))
y_mean_errors = np.zeros((len(datasets), total_epochs))

for epoch in range(total_epochs):
    #Reset the environment
    step = 0
    done = False
    total_rewards = np.zeros(2)

    for set in range(len(datasets)-1):

        for step in range(state_size-1):
            #Choose an action a in the current world state s, can be exploitation or exploration
            ## First we randomise a number
            exp_exp_tradeoff = random.uniform(0,1)

            ##If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            if exp_exp_tradeoff > epsilon:
                x_action = np.argmax(qtable[0,step, :])
                y_action = np.argmax(qtable[1,step, :])

            #Else doing a random choice --> exploration
            else:
                x_action = np.random.randint(action_size)
                y_action = np.random.randint(action_size)

            #Take the action a and observe the outcome state s' and reward r
            datasets[set].xfbk[step] += actions[x_action]*np.sign(datasets[set].xcom[step]-datasets[set].xfbk[step]) #add 1, 0 or -1 to demonstrate accel, const or decel
            datasets[set].yfbk[step] += actions[y_action]*np.sign(datasets[set].ycom[step]-datasets[set].yfbk[step])
            datasets[set].errors[0,step] = np.absolute(datasets[set].xcom[step]-datasets[set].xfbk[step])
            datasets[set].errors[1,step] = np.absolute(datasets[set].ycom[step]-datasets[set].yfbk[step])
            datasets[set].errors[0,step+1] = np.absolute(datasets[set].xcom[step+1]-datasets[set].xfbk[step+1])
            datasets[set].errors[1,step+1] = np.absolute(datasets[set].ycom[step+1]-datasets[set].yfbk[step+1])
            x_reward = check_error(datasets[set].errors[0,step], datasets[set].errors[0,step+1])
            y_reward = check_error(datasets[set].errors[1,step], datasets[set].errors[1,step+1])


            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            #qtable[0, step, x_action] = qtable[0, step, x_action] + learning_rate*(x_reward + gamma * np.max(qtable[0,step+1, :]) - qtable[0,step,x_action])
            #qtable[1, step, y_action] = qtable[1, step, y_action] + learning_rate*(y_reward + gamma * np.max(qtable[0,step+1, :]) - qtable[0,step,y_action])

            #x_qtable[step,x_action] = x_qtable[step, x_action] + learning_rate*(x_reward + gamma * np.max(x_qtable[step+1, :]) - x_qtable[step,x_action])
            #y_qtable[step,y_action] = y_qtable[step, y_action] + learning_rate*(y_reward + gamma * np.max(y_qtable[step+1, :]) - y_qtable[step,y_action])

            total_rewards += (x_reward, y_reward)

        x_mean_step = np.mean(np.absolute(datasets[set].errors[0,:]))
        y_mean_step = np.mean(np.absolute(datasets[set].errors[1,:]))
        datasets[set].mean_errors.append([x_mean_step,y_mean_step])
        x_mean_errors[set, epoch] = x_mean_step
        y_mean_errors[set, epoch] = y_mean_step
        delta_err = np.subtract(datasets[set].init_mean, (x_mean_step, y_mean_step))
        #print("Initial mean error for dataset {} was: {} In this epoch, ({}), Q-Learning has changed this by {} to: {} ".format(set,datasets[set].init_mean,epoch, delta_err, n_mean_step))


    overall_mean = (np.mean(x_mean_errors[:, epoch]),np.mean(y_mean_errors[:, epoch]))
    prev_mean = (np.mean(x_mean_errors[:, epoch-1]),np.mean(y_mean_errors[:, epoch-1]))
    init_delta = np.absolute(np.subtract(overall_init_mean,overall_mean))
    epoch_delta = np.absolute(np.subtract(overall_mean,prev_mean))
    print("Initial mean error was: {}, reduced in epoch {} to {}. A change of {} from previous, {} from init.".format(overall_init_mean,epoch, overall_mean, epoch_delta, init_delta))
    #Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon +(max_epsilon - min_epsilon)*np.exp(-decay_rate*epoch)
    epoch += 1
    rewards.append(total_rewards)

end_mean = []
for i in range(0,len(epochs)-1):
    end_mean.append(np.mean(mean_errors[:,i]))
plt.plot(end_mean)
plt.ylabel('mean error per epoch')
plt.xlabel('Number of epochs')
plt.show()
