#Test of my parser working in a 'main' file

import os
import tensorflow as tf
import numpy as np
import pandas as pd
from parser import parser

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
    datasets.append(dataset(temp))

max_steps = 0
for set in datasets:
    if set.num_steps > max_steps:
        max_steps = set.num_steps

for set in datasets:
    set.pad_size(max_steps)

actions = [1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0, -.1, -.2, -.3, -.4, -.5, -.6, -.7, -.8, -.9, -1]
a_size = len(actions)

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
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = self.optimizer.minimize(self.loss)

#Now move on to training the agent, for which we need to actually invoke a tensorflow session
total_episodes = 100   #Set total number of episodes to train the agent on. IRL this will be training on as much data as I can give it.
total_reward = np.zeros(a_size)   #Set scoreboard for bandits to zero.
e = 0.6 #Set the chance of taking a random action

mean_errors = np.zeros((len(datasets), total_episodes))

#Now move on to establising the agent
tf.reset_default_graph()

#Initialise the agent.
the_agent = agent(lr = 0.001, s_size = max_steps, a_size = a_size)

init = tf.global_variables_initializer()

#Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    for k in range(total_episodes):
        for s in range(len(datasets)):
            for i in range(max_steps-1):
                #Choose either a random action or onefrom our network.
                if np.random.rand(1) < e:
                    action = np.random.randint(a_size)
                else:
                    action = sess.run(the_agent.chosen_action)


                datasets[s].xfbk[i-1] += actions[action]*np.sign(datasets[s].xcom[i-1]-datasets[s].xfbk[i-1]) #add 1, 0 or -1 to demonstrate accel, const or decel
                datasets[s].errors[i-1] = np.absolute(datasets[s].xcom[i-1]-datasets[s].xfbk[i-1])
                datasets[s].errors[i] = np.absolute(datasets[s].xcom[i]-datasets[s].xfbk[i])
                reward = check_error(datasets[s].errors[i-1], datasets[s].errors[i])

                #Update the network.
                _,resp,ww = sess.run([the_agent.update,the_agent.responsible_weight,the_agent.weights], feed_dict={the_agent.reward_holder:[reward],the_agent.action_holder:[action]})
                #print(str(action)+" : "+str(reward))

                #Update running tally of scores
                total_reward[action] += reward
                '''if i % 1000 == 0:
                    print("Running reward: "+str(total_reward))
                    print("xcommand is: "+str(xcom[i])+" and xfeedback: "+str(xfbk[i]))'''

            n_mean_step = np.mean(np.absolute(datasets[s].errors))
            datasets[s].mean_errors.append(n_mean_step)
            mean_errors[s, k] = n_mean_step
            delta_err = datasets[s].init_mean - n_mean_step
            print("Initial mean error for dataset {} was: {} In this episode, ({}), Reinforcement Learning has changed this by {} to: {} ".format(s,datasets[s].init_mean, k, delta_err, n_mean_step))

plt.plot(mean_errors)
plt.ylabel('mean error per iteration')
plt.xlabel('Number of iterations')
plt.show()
