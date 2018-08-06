import random
import os
import numpy as np
from parse import parser
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

EPISODES = 1000


def chunk_list(list, n):
    for i in range(0, len(list), n):
        yield list[i: i + n]


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


class dataset():
    def __init__(self, data):
        self.xcom = data[:, 0]
        self.ycom = data[:, 1]
        self.xfbk = data[:, 2]
        self.yfbk = data[:, 3]
        self.init_mean = (np.mean(np.absolute(self.xcom-self.xfbk)))
        self.num_steps = len(self.xcom) # num_steps is constant for x and y
        self.errors = np.zeros(self.num_steps)
        self.mean_errors = []

    def pad_size(self, max_steps):
        difference = max_steps - self.num_steps
        filler = np.zeros(difference)
        self.num_steps = max_steps
        self.xcom = np.concatenate((self.xcom, filler))
        self.xfbk = np.concatenate((self.xfbk, filler))
        self.ycom = np.concatenate((self.ycom, filler))
        self.yfbk = np.concatenate((self.yfbk, filler))
        self.errors = np.zeros(self.num_steps)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":

    actions = np.linspace(-10, 10, 201)
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
    mean_errors = np.zeros((len(train_sets), EPISODES))

    overall_init_mean = np.zeros(2)
    sum = np.zeros(2)
    for set in train_sets:
        for i in range(0, 1):
            sum += set.init_mean
    overall_init_mean = sum/len(train_sets)

    with tf.device('/gpu:0'):
        agent = DQNAgent(state_size, action_size)
        # agent.load("./save/cartpole-dqn.h5")
        done = False
        batch_size = 32

        for e in range(EPISODES):

            temp_xfbk = np.zeros(state_size)
            temp_yfbk = np.zeros(state_size)
            temp_err = np.zeros(state_size)
            state = train_sets[0].errors
            state = np.reshape(state, [1, state_size])
            for set in range(len(train_sets)-1):

                for step in range(state_size-1):
                    # env.render()
                    action = agent.act(state)
                    # next_state, reward, done, _ = env.step(action)
                    temp_xfbk[step] = train_sets[set].xfbk[step] + actions[action]*np.sign(train_sets[set].xcom[step]-train_sets[set].xfbk[step]) # add 1, 0 or -1 to demonstrate accel, const or decel
                    temp_err[step] = np.absolute(train_sets[set].xcom[step]-temp_xfbk[step])
                    temp_err[step+1] = np.absolute(train_sets[set].xcom[step+1]-train_sets[set].xfbk[step+1])
                    reward = check_error(temp_err[step], temp_err[step+1])

                next_state = state
                next_state = np.reshape(next_state, [1, state_size])
                if step == state_size - 1:
                    done = True
                else:
                    done = False
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

                train_sets[set].errors = temp_err
                n_mean_step = np.mean(np.absolute(train_sets[set].errors))
                train_sets[set].mean_errors.append(n_mean_step)
                mean_errors[set, e] = n_mean_step
                delta_err = np.subtract(train_sets[set].init_mean, n_mean_step)
                # print("Initial mean error for dataset {} was: {} In this epoch, ({}), Q-Learning has changed this by {} to: {} ".format(set,train_sets[set].init_mean,epoch, delta_err, n_mean_step))

            overall_mean = np.mean(mean_errors[:, e])
            prev_mean = np.mean(mean_errors[:, e-1])
            init_delta = np.absolute(overall_init_mean - overall_mean)
            epoch_delta = np.absolute(prev_mean - overall_mean)
            print("Initial mean error was: {}, reduced in epoch {} to {}. A change of {} from previous, {} from init.".format(overall_init_mean, e, overall_mean, epoch_delta, init_delta))

            # if e % 10 == 0:
    #     agent.save("./save/cartpole-dqn.h5")
