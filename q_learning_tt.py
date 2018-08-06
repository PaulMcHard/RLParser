import os
import numpy as np
import random
import pandas as pd
from parse import parser
import matplotlib.pyplot as plt
# import keras


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
        self.num_steps = len(self.xcom) # num_steps is constant for x and y
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
        self.errors = np.zeros((2,self.num_steps))


actions = [2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0, -.1, -.2, -.3, -.4, -.5, -.6, -.7, -.8, -.9, -1, -1.1, -1.2, -1.3 -1.4, -1.5, -1.6, -1.7, -1.8, -1.9, -2]
action_size = len(actions)
dirname = os.getcwd()
dirname += '/data/'
data_parser = parser()
files = []
for file in os.listdir(dirname):
    if file.endswith(".DAT"):
        files.append(file)

chosen_test_set = 0
file_data_train = []
file_data_test = []
for i in range(len(files)):
    data_parser.parse_data(dirname+files[i])
    temp = data_parser.get_all_com_fbk().values
    if i == chosen_test_set:
        file_data_test.append(temp)
    else:
        file_data_train.append(temp)

set_length = 500
train_sets = []
test_sets = []
for file in file_data_train:
    sets = list(chunk_list(file, set_length))
    for set in sets:
        temp = dataset(set)
        if temp.num_steps < set_length:
            temp.pad_size(set_length)
        train_sets.append(temp)

random.shuffle(train_sets)

for file in file_data_test:
    sets = list(chunk_list(file, set_length))
    for set in sets:
        temp = dataset(set)
        if temp.num_steps < set_length:
            temp.pad_size(set_length)
        test_sets.append(temp)

state_size = set_length
# qtable = np.zeros((state_size, action_size))
x_qtable = np.zeros((state_size, action_size))
y_qtable = np.zeros((state_size, action_size))


# Function below gives our reward system.
def check_error(e_n, e_nplus):
    # Check error subtracted from previous error
    result = e_n - e_nplus
    if result < 0:
        # return a positive reward.
        return 100
    elif result == 0:
        # return a neutral reward.
        return 0
    else:
        # eturn a negative reward.
        return -1


# SPECIFY HYPERPARAMETERS
total_epochs = 1000       # Total total_epochs
learning_rate = 0.003          # Learning Rate
gamma = 0.95                 # Discounting Rate

# Exploration parameters
epsilon = 1.0       # Exploration Rate
max_epsilon = 1.0   # Exploration Probability at start
min_epsilon = 0.01  # Minimum exploration Rate
decay_rate = 0.01   # Exponential decayrate for exploration prob

# List of rewards
rewards = []

overall_init_mean = np.zeros(2)
sum = np.zeros(2)
for set in train_sets:
    for i in range(0,1):
        sum += set.init_mean
overall_init_mean = sum/len(train_sets)

x_mean_errors = np.zeros((len(train_sets), total_epochs))
y_mean_errors = np.zeros((len(train_sets), total_epochs))

# TRAIN QTABLE
for epoch in range(total_epochs):
    # Reset the q table
    step = 0
    total_rewards = np.zeros(2)

    for set in range(len(train_sets)-1):

        for step in range(state_size-1):
            # Choose an action a in the current world state s, can be exploitation or exploration
            # #First we randomise a number
            exp_exp_tradeoff = random.uniform(0, 1)

            # #If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            if exp_exp_tradeoff > epsilon:
                x_action = np.argmax(x_qtable[step, :])
                y_action = np.argmax(y_qtable[step, :])

            # Else doing a random choice --> exploration
            else:
                x_action = np.random.randint(action_size)
                y_action = np.random.randint(action_size)

            # Take the action a and observe the outcome state s' and reward r
            train_sets[set].xfbk[step] += actions[x_action]*np.sign(train_sets[set].xcom[step]-train_sets[set].xfbk[step]) # add 1, 0 or -1 to demonstrate accel, const or decel
            train_sets[set].yfbk[step] += actions[y_action]*np.sign(train_sets[set].ycom[step]-train_sets[set].yfbk[step])
            train_sets[set].errors[0,step] = np.absolute(train_sets[set].xcom[step]-train_sets[set].xfbk[step])
            train_sets[set].errors[1,step] = np.absolute(train_sets[set].ycom[step]-train_sets[set].yfbk[step])
            train_sets[set].errors[0,step+1] = np.absolute(train_sets[set].xcom[step+1]-train_sets[set].xfbk[step+1])
            train_sets[set].errors[1,step+1] = np.absolute(train_sets[set].ycom[step+1]-train_sets[set].yfbk[step+1])
            x_reward = check_error(train_sets[set].errors[0,step], train_sets[set].errors[0,step+1])
            y_reward = check_error(train_sets[set].errors[1,step], train_sets[set].errors[1,step+1])


            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            # qtable[0, step, x_action] = qtable[0, step, x_action] + learning_rate*(x_reward + gamma * np.max(qtable[0,step+1, :]) - qtable[0,step,x_action])
            # qtable[1, step, y_action] = qtable[1, step, y_action] + learning_rate*(y_reward + gamma * np.max(qtable[0,step+1, :]) - qtable[0,step,y_action])

            x_qtable[step, x_action] = x_qtable[step, x_action] + learning_rate * (x_reward + gamma * np.max(x_qtable[step+1, :]) - x_qtable[step,x_action])
            y_qtable[step, y_action] = y_qtable[step, y_action] + learning_rate * (y_reward + gamma * np.max(y_qtable[step+1, :]) - y_qtable[step,y_action])

            total_rewards += (x_reward, y_reward)

        x_mean_step = np.mean(np.absolute(train_sets[set].errors[0,:]))
        y_mean_step = np.mean(np.absolute(train_sets[set].errors[1,:]))
        train_sets[set].mean_errors.append([x_mean_step,y_mean_step])
        x_mean_errors[set, epoch] = x_mean_step
        y_mean_errors[set, epoch] = y_mean_step
        delta_err = np.subtract(train_sets[set].init_mean, (x_mean_step, y_mean_step))
        # print("Initial mean error for dataset {} was: {} In this epoch, ({}), Q-Learning has changed this by {} to: {} ".format(set,train_sets[set].init_mean,epoch, delta_err, n_mean_step))

    overall_mean = (np.mean(x_mean_errors[:, epoch]), np.mean(y_mean_errors[:, epoch]))
    prev_mean = (np.mean(x_mean_errors[:, epoch-1]), np.mean(y_mean_errors[:, epoch-1]))
    init_delta = np.absolute(np.subtract(overall_init_mean,overall_mean))
    epoch_delta = np.absolute(np.subtract(overall_mean,prev_mean))
    print("Initial mean error was: {}, reduced in epoch {} to {}. A change of {} from previous, {} from init.".format(overall_init_mean,epoch + 1, overall_mean, epoch_delta, init_delta))
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*epoch)
    epoch += 1
    rewards.append(total_rewards)

# SAVE MODEL
model_name_x = "QTABLE_X_TESTON{}".format(chosen_test_set)
model_name_y = "QTABLE_Y_TESTON{}".format(chosen_test_set)
np.savetxt(model_name_x, x_qtable)
np.savetxt(model_name_y, y_qtable)

# PLOT TRAINING RESULTS
x_end_mean = []
y_end_mean = []
X = np.linspace(0, total_epochs, total_epochs)
for i in range(0, total_epochs):
    x_end_mean.append(np.mean(x_mean_errors[:, i]))
    y_end_mean.append(np.mean(y_mean_errors[:, i]))
plt.plot(x_end_mean, color='k', label='Error in X')
plt.plot(y_end_mean, color='g', label='Error in Y')
plt.legend()
plt.ylabel('mean error per epoch')
plt.xlabel('Number of epochs')
plt.title("Q Learning method with test on file {} .".format(chosen_test_set))
plt.savefig("PLOT_QTABLE_TESTON{}".format(chosen_test_set))

test_init_mean = np.zeros(2)
t_sum = np.zeros(2)
for set in test_sets:
    for i in range(0, 1):
        t_sum += set.init_mean
test_init_mean = t_sum/len(test_sets)

test_x_mean_errors = np.zeros(len(test_sets))
test_y_mean_errors = np.zeros(len(test_sets))

#TEST MODEL
for set in range(len(test_sets)-1):

    for step in range(state_size-1):
        x_action = np.argmax(x_qtable[step, :])
        y_action = np.argmax(y_qtable[step, :])

        test_sets[set].xfbk[step] += actions[x_action] * np.sign(test_sets[set].xcom[step] - test_sets[set].xfbk[step]) # add 1, 0 or -1 to demonstrate accel, const or decel
        test_sets[set].yfbk[step] += actions[y_action] * np.sign(test_sets[set].ycom[step] - test_sets[set].yfbk[step])
        test_sets[set].errors[0, step] = np.absolute(test_sets[set].xcom[step]- test_sets[set].xfbk[step])
        test_sets[set].errors[1, step] = np.absolute(test_sets[set].ycom[step]- test_sets[set].yfbk[step])
        test_sets[set].errors[0, step+1] = np.absolute(test_sets[set].xcom[step + 1] - test_sets[set].xfbk[step + 1])
        test_sets[set].errors[1, step+1] = np.absolute(test_sets[set].ycom[step + 1] - test_sets[set].yfbk[step + 1])

    tx_mean_step = np.mean(np.absolute(test_sets[set].errors[0, :]))
    ty_mean_step = np.mean(np.absolute(test_sets[set].errors[1, :]))
    test_x_mean_errors[set] = tx_mean_step
    test_y_mean_errors[set] = ty_mean_step

test_end_mean = (np.mean(test_x_mean_errors[:]), np.mean(test_y_mean_errors[:]))

print("Test mean changed from {} to {}".format(test_init_mean, test_end_mean))

# SHOW PLOT AFTER TESTING
plt.show()
