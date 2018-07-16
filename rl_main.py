#Test of my parser working in a 'main' file
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from parser import parser

dirname = os.getcwd()
data_parser = parser()
data_parser.parse_data(dirname+'/data/03cadTEST3.DAT')
x = data_parser.get_x().values
xcom = x[:,0]
xfbk = x[:,1]
o_mean = np.mean(np.absolute(xcom-xfbk))
num_steps = len(xcom)
actions = [1,0.5,0.1,0,-0.1,-0.5, -1]
a_size = len(actions)
errors = np.zeros(num_steps)

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
'''
errors[0] = xcom[0]-xfbk[0]
for i in range(1:num_steps):
    errors[i] = xcom[i]-xfbk[i]
    if check_error
'''
#Now move on to establising the agent
tf.reset_default_graph()

#The next two lines establish the feed-forward part of the network. This does the actual choosing.
weights = tf.Variable(tf.ones([a_size]))
chosen_action = tf.argmax(weights,0)

#The next set of lines establish the training procedure. We feed the reward and chosen action into the Network
#to compute the loss, and use it to update the network.
reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
responsible_weight = tf.slice(weights,action_holder,[1])
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

#Now move on to training the agent, for which we need to actually invoke a tensorflow session
total_episodes = 100   #Set total number of episodes to train the agent on. IRL this will be training on as much data as I can give it.
total_reward = np.zeros(a_size)   #Set scoreboard for bandits to zero.
e = 0.6 #Set the chance of taking a random action


init = tf.global_variables_initializer()

#Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    k=0
    while k < total_episodes:
        i = 0
        errors[0]= np.absolute(xcom[0]-xfbk[0])
        while i <(num_steps-1):
            i += 1
            #Choose either a random action or onefrom our network.
            if np.random.rand(1) < e:
                action = np.random.randint(a_size)
            else:
                action = sess.run(chosen_action)
            xfbk[i-1] += actions[action]*np.sign(xcom[i-1]-xfbk[i-1]) #add 1, 0 or -1 to demonstrate accel, const or decel
            errors[i-1] = np.absolute(xcom[i-1]-xfbk[i-1])
            errors[i] = np.absolute(xcom[i]-xfbk[i])
            reward = check_error(errors[i-1], errors[i]) #Get our reward from picking one of the bandits.

            #Update the network.
            _,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward],action_holder:[action]})
            #print(str(action)+" : "+str(reward))

            #Update running tally of scores
            total_reward[action] += reward
            if i % 1000 == 0:
                print("Running reward: "+str(total_reward))
                print("xcommand is: "+str(xcom[i])+" and xfeedback: "+str(xfbk[i]))
        n_mean_step = np.mean(np.absolute(errors))
        print("The original mean error was: "+str(o_mean)+" RL  this step has changed this to: "+str(n_mean_step))
        k += 1

    n_mean = np.mean(np.absolute(errors))
    print("The original mean error was: "+str(o_mean)+" RL has changed this to: "+str(n_mean))
