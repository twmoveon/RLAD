from DQN import DeepQNetwork
from Codes.RL4AD.environment.time_series_env import TimeSeriesEnv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
DATAFIXED = 0           # whether target at a single time series dataset
# Reward Values
TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5

NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

n_steps = 25        # size of the slide window for SLIDE_WINDOW state and reward functions
n_input_dim = 2     # dimension of the input for a LSTM cell
n_hidden_dim = 64   # dimension of the hidden state in LSTM cell


def RNNBinaryStateFuc(timeseries, timeseries_curser, previous_state=[], action=None):
    if timeseries_curser == n_steps:
        state = []
        for i in range(timeseries_curser):
            state.append([timeseries['value'][i], 0])

        state.pop(0)
        state.append([timeseries['value'][timeseries_curser], 1])

        return np.array(state, dtype='float32')

    if timeseries_curser > n_steps:
        state0 = np.concatenate((previous_state[1:n_steps],
                                 [[timeseries['value'][timeseries_curser], 0]]))
        state1 = np.concatenate((previous_state[1:n_steps],
                                 [[timeseries['value'][timeseries_curser], 1]]))

        return np.array([state0, state1], dtype='float32')


# Also, because we use binary tree here, the reward function returns a list of rewards for each action
def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0):
    if timeseries_curser >= n_steps:
        if timeseries['anomaly'][timeseries_curser] == 0:
            return [TN_Value, FP_Value]

        if timeseries['anomaly'][timeseries_curser] == 1:
            return [FN_Value, TP_Value]
    else:
        return [0, 0]


# set evn
env = TimeSeriesEnv()
env.state_function = RNNBinaryStateFuc
env.reward_function= RNNBinaryRewardFuc
env.cursor_init = n_steps
env.datasetfix = DATAFIXED
env.index = 0

for i in range(20):
    print env.reset()
'''
# set RL method
RL = DeepQNetwork(
                  learning_rate=0.01, e_greedy=0.5,
                  replace_target_iter=10000, memory_size=500000,
                  e_greedy_increment=0.001, batch_size=256)

total_steps = 0
steps = []
# start training
for i_episode in range(2000):

    observation = environment.reset()
    ep_r = 0
    step = 0
    while True:
        step = step + 1

        action = RL.choose_action(observation)  # choose an action

        observation_, reward, done, info = environment.step(action)     # observation from environment

        RL.store_transition(observation, action, reward, observation_)      # store them into memory: off policy
        print reward
        ep_r += reward
        if total_steps > 50000:      # store some memory before learning
            RL.learn()

        if done:
            steps.append(step)
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            print observation
            break

        observation = observation_
        total_steps += 1
'''
