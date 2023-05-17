import pandas as pd
import numpy as np
import os
import random
import sklearn.preprocessing

TIME_SERIES_PATH = '/Users/wuxiaodong/Downloads/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/'
ANOMALY = 1
NOT_ANOMALY = 0
ACTION_SPACE = [ANOMALY, NOT_ANOMALY]
REWARD_CORRECT = 1
REWARD_INCORRECT = -1


def read_csv():
    subdir_list = []
    dir_list = os.listdir(TIME_SERIES_PATH)
    for dir in dir_list:
        subdir_list.append(os.path.join(TIME_SERIES_PATH, dir))
    return subdir_list


def default_state_function(time_series, cursor):
    return time_series['value'][cursor]


def default_reward_function(time_series, cursor, action):
    if action == time_series['anomaly'][cursor]:
        return REWARD_CORRECT
    else:
        return REWARD_INCORRECT


class TimeSeriesEnv():
    def __init__(self):
        self.action_space = len(ACTION_SPACE)
        self.time_series = []
        self.cursor = -1
        self.cursor_init = 0
        self.dir_list = read_csv()
        self.state_function = default_state_function
        self.reward_function = default_reward_function

        self.datasetfix = 0
        self.index = random.randint(0, len(self.dir_list) - 1)

        self.dataset_size = len(self.dir_list)
        self.timeseries_repo = []

        for i in range(len(self.dir_list)):
            """
            The following two lines are used instead of the third line when DataMarket is the data source.
            """
            # ts = pd.read_csv(self.repodirext[i], usecols=[1], header=0, skipfooter=2, names=['value'], engine='python')
            # ts['anomaly'] = pd.Series(np.zeros(len(ts['value'])), index=ts.index)

            """
            The following line is used instead of the third line when Numenta is the data source.
            """
            #ts = pd.read_csv(self.dir_list[i], usecols=[1, 3], header=0, names=['value', 'anomaly'])

            """
            The following line is used instead of the third line when Yahoo Benchmark is the data source.
            """
            ts = pd.read_csv(self.dir_list[i], usecols=[1,2], header=0, names=['value','anomaly'])

            ts = ts.astype(np.float32)

            scaler = sklearn.preprocessing.MinMaxScaler()
            scaler.fit(np.array(ts['value']).reshape(-1, 1))
            ts['value'] = scaler.transform(np.array(ts['value']).reshape(-1, 1))

            self.timeseries_repo.append(ts)

    def reset(self):
        if self.datasetfix == 0:
            self.index = random.randint(0, self.dataset_size - 1)
        self.time_series = self.timeseries_repo[self.index]
        self.cursor = self.cursor_init

        state = self.state_function(self.time_series, self.cursor)
        return state

    def step(self, action):
        assert action in ACTION_SPACE
        assert self.cursor >= 0

        reward = self.reward_function(self.time_series, self.cursor, action)
        self.cursor += 1

        if self.cursor >= self.time_series['value'].size:
            done = 1
            state = []
        else:
            done = 0
            state = self.state_function(self.time_series, self.cursor)
        return state, reward, done, []
