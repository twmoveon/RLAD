import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import sys
import os
import time

import tensorflow as tf

from tensorflow.contrib import rnn
from mpl_toolkits.mplot3d import axes3d
from collections import deque, namedtuple

# import the library in the sub-folder environment
from Codes.RL4AD.environment.time_series_repo_ext import EnvTimeSeriesfromRepo


# macros for running q-learning
DATAFIXED = 0           # whether target at a single time series dataset

EPISODES = 500          # number of episodes for training
DISCOUNT_FACTOR = 0.5   # reward discount factor [0,1]
EPSILON = 0.5           # epsilon-greedy method parameter for action selection
EPSILON_DECAY = 1.00    # epsilon-greedy method decay parameter

NOT_ANOMALY = 0
ANOMALY = 1

action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

n_steps = 25        # size of the slide window for SLIDE_WINDOW state and reward functions
n_input_dim = 2     # dimension of the input for a LSTM cell
n_hidden_dim = 64   # dimension of the hidden state in LSTM cell

# Reward Values
TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5

validation_separate_ratio = 0.9


# The state function returns a vector composing of n_steps of n_input_dim data instances:
# e.g., [[x_1, f_1], [x_2, f_2], ..., [x_t, f_t]] of shape (n_steps, n_input_dim)
# x_t is the new data instance. t here is equal to n_steps

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


class Q_Estimator_Nonlinear():
    """
    Action-Value Function Approximator Q(s,a) with Tensorflow RNN.
    Note: The Recurrent Neural Network is used here !
    """

    def __init__(self, learning_rate=np.float32(0.01), scope="Q_Estimator_Nonlinear", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None

        with tf.variable_scope(scope):
            # tf Graph input
            # The input to the rnn is typically of the shape:
            # [batch_size, n_steps, n_input_dim]
            # RNN requires the data of the shape:
            # n_steps tensors of [batch_size, n_input_dim]
            self.state = tf.placeholder(shape=[None, n_steps, n_input_dim],
                                        dtype=tf.float32, name="state")
            self.target = tf.placeholder(shape=[None, action_space_n],
                                         dtype=tf.float32, name="target")

            # Define weights
            self.weights = {
                'out': tf.Variable(tf.random_normal([n_hidden_dim, action_space_n]))
            }
            self.biases = {
                'out': tf.Variable(tf.random_normal([action_space_n]))
            }

            self.state_unstack = tf.unstack(self.state, n_steps, 1)
            print self.state_unstack

            # Define a lstm cell with tensorflow
            lstm_cell = rnn.BasicLSTMCell(n_hidden_dim, forget_bias=1.0)

            # Get lstm cell output
            self.outputs, self.states = rnn.static_rnn(lstm_cell,
                                                       self.state_unstack,
                                                       dtype=tf.float32)

            # Linear activation, using rnn inner loop last output
            self.action_values = tf.matmul(self.outputs[-1], self.weights['out']) + self.biases['out']

            # Loss and train op
            self.losses = tf.squared_difference(self.action_values, self.target)
            self.loss = tf.reduce_mean(self.losses)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss,
                                                    global_step=tf.contrib.framework.get_global_step())

            # Summaries for Tensorboard
            self.summaries = tf.summary.merge([
                tf.summary.histogram("loss_hist", self.losses),
                tf.summary.scalar("loss", self.loss),
                tf.summary.histogram("q_values_hist", self.action_values),
                tf.summary.scalar("q_value", tf.reduce_max(self.action_values))
            ])

            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_values, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        summaries, global_step, _ = sess.run([self.summaries, tf.contrib.framework.get_global_step(),
                                              self.train_op], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action. float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype='float32') * epsilon / nA
        q_values = estimator.predict(state=[observation])
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env,
               qlearn_estimator,
               target_estimator,
               num_episodes,
               num_epoches,
               replay_memory_size=500000,
               replay_memory_init_size=50000,
               experiment_dir='./log/',
               update_target_estimator_every=10000,
               discount_factor=0.99,
               epsilon_start=1.0,
               epsilon_end=0.1,
               epsilon_decay_steps=500000,
               batch_size=512,
               test=0):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: The environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        NULL
    """
    # 1. Define some useful variable
    # To memory
    Transition = namedtuple("Transition", ["state", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    checkpoint_path = os.path.join(checkpoint_dir, "model")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load a previous checkpoint if we find one
    saver = tf.train.Saver()

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
        if test:
            return

    # Get the time step
    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        qlearn_estimator,
        env.action_space_n)

    # 2. Populate the replay memory with initial experience
    popu_time = time.time()

    state = env.reset()
    while env.datasetidx > env.datasetsize * validation_separate_ratio:
        env.reset()
        print 'double reset'

    for i in range(replay_memory_init_size):
        action_probs = policy(state, epsilons[min(total_t, epsilon_decay_steps - 1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        next_state, reward, done, _ = env.step(action)

        replay_memory.append(Transition(state, reward, next_state, done))

        if done:
            state = env.reset()
            while env.datasetidx > env.datasetsize * validation_separate_ratio:
                env.reset()
                print 'double reset'
        else:
            state = next_state[action]

    popu_time = time.time() - popu_time
    print("Populating replay memory with time {}".format(popu_time))

    # 3. Start the main loop
    for i_episode in range(num_episodes):
        # Save the current checkpoint
        if i_episode % 50 == 49:
            print("Save checkpoint in episode {}/{}".format(i_episode + 1, num_episodes))
            saver.save(tf.get_default_session(), checkpoint_path)

        per_loop_time1 = time.time()

        # Reset the environment
        state = env.reset()
        while env.datasetidx > env.datasetrng * validation_separate_ratio:
            env.reset()
            print 'double reset'

        # One step in the environment
        for t in itertools.count():

            # 3.1 Some Preprocess
            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]

            # 3.2 The main work
            # Choose an action to take
            action_probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # Take a step
            next_state, reward, done, _ = env.step(action)

            # 3.3 Control replay memory
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            replay_memory.append(Transition(state, reward, next_state, done))

            if done:
                break

            state = next_state[action]

        per_loop_time2 = time.time()

        # Update the model
        for i_epoch in range(num_epoches):
            # Add epsilon to Tensorboard
            if qlearn_estimator.summary_writer:
                episode_summary = tf.Summary()
                qlearn_estimator.summary_writer.add_summary(episode_summary, total_t)

            # Update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, qlearn_estimator, target_estimator)
                print("\nCopied model parameters to target network.\n")

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            if discount_factor > 0:
                # Calculate q values and targets
                next_states_batch = np.squeeze(np.split(next_states_batch, 2, axis=1))
                next_states_batch0 = next_states_batch[0]
                next_states_batch1 = next_states_batch[1]

                q_values_next0 = target_estimator.predict(state=next_states_batch0)
                q_values_next1 = target_estimator.predict(state=next_states_batch1)

                targets_batch = reward_batch + (discount_factor *
                                                np.stack((np.amax(q_values_next0, axis=1),
                                                          np.amax(q_values_next1, axis=1)),
                                                         axis=-1))
            else:
                targets_batch = reward_batch

            # Perform gradient descent update
            qlearn_estimator.update(state=states_batch, target=targets_batch.astype(np.float32))

            total_t += 1

        # Print out which step we're on, useful for debugging.
        per_loop_time_popu = per_loop_time2 - per_loop_time1
        per_loop_time_updt = time.time() - per_loop_time2
        print("Global step {} @ Episode {}/{}, time: {} + {}"
              .format(total_t, i_episode + 1, num_episodes, per_loop_time_popu, per_loop_time_updt))
    return


def q_learning_validator(env, estimator, num_episodes, record_dir=None, plot=1):
    """
    With 1) the trained estimator of Q(s,a) action-value function, i.e., estimator
         2) the known envivronment, i.e., environment

    Test the validity of the estimator in the application of time series anomaly detection.

    Args:
        env: The environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.

    Returns:
        Statistics
    """
    if record_dir:
        rec_file = open('./{}'.format(record_dir), 'wb')
        rec_writer = csv.writer(rec_file)

    p_overall = 0
    recall_overall = 0
    f1_overall = 0

    for i_episode in range(num_episodes):
        print("Episode {}/{}".format(i_episode + 1, num_episodes))

        state_rec = []
        action_rec = []
        reward_rec = []

        # The policy we're following
        policy = make_epsilon_greedy_policy(
            estimator, env.action_space_n)

        # Reset the environment and pick the first action
        state = env.reset()
        while env.datasetidx < env.datasetrng * validation_separate_ratio:
            state = env.reset()
            print 'double reset'
        print 'testing on: ' + str(env.repodirext[env.datasetidx])
        # One step in the environment
        for t in itertools.count():
            # Choose an action to take
            action_probs = policy(state, 0)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # Take a step
            next_state, reward, done, _ = env.step(action)

            # Record the statistics
            state_rec.append(state[len(state) - 1][0])
            action_rec.append(action)
            reward_rec.append(reward[action])

            if done:
                break

            state = next_state[action]

        # Search over reward_rec to correct the rewards
        # 1) If FN happend within a small range of TP (miss alarm very closer to reported anomaly), correct it
        # 2) If FP happend within a small range of TP (false alarm very close to the reported anomaly), correct it

        RNG = 5
        for i in range(len(reward_rec)):
            if reward_rec[i] < 0:
                low_range = max(0, i - RNG)
                up_range = min(i + RNG + 1, len(reward_rec))
                r = reward_rec[low_range:up_range]
                if r.count(TP_Value) > 0:
                    reward_rec[i] = -reward_rec[i]

        # Plot the result for each episode
        if plot:
            f, axarr = plt.subplots(3, sharex=True)
            axarr[0].plot(state_rec)
            axarr[0].set_title('Time Series')
            axarr[1].plot(action_rec, color='g')
            axarr[1].set_title('Action')
            axarr[2].plot(reward_rec, color='r')
            axarr[2].set_title('Reward')
            plt.show()
            #plt.savefig("/home/lfwutong/sbsplusplus/graphs/Aniyama-dataport/"+str(i_episode)+".png")
            #plt.savefig("/Users/wuxiaodong/Dropbox/adaptive-anomalies/demo/graphs/Aniyama-dataport/"+str(i_episode)+".png")

        # Calculate the accuracy F1-score = 2*((precison*recall)/(precison+recall))
        # precision = tp / (tp+fp)
        # recall = tp / (tp+fn)

        tp = reward_rec.count(TP_Value)
        fp = reward_rec.count(FP_Value)
        fn = reward_rec.count(FN_Value)
        precision = (tp + 1) / float(tp + fp + 1)
        recall = (tp + 1) / float(tp + fn + 1)
        f1 = 2 * ((precision * recall) / (precision + recall))
        p_overall += precision
        recall_overall += recall
        f1_overall += f1

        if record_dir:
            rec_writer.writerow([f1])

        print("Precision:{}, Recall:{}, F1-score:{} (f1 wrote to file)".format(precision, recall, f1))
    print ("Precision:{}, Recall:{}, F1-score:{} ".
           format(p_overall / num_episodes, recall_overall / num_episodes, f1_overall / num_episodes))
    print ("Precision:{}, Recall:{}, F1-score:{} ".
                   format(p_overall / num_episodes, recall_overall / num_episodes, f1_overall / num_episodes))
    if record_dir:
        rec_file.close()

    return


# percentage = ['0.2', '0.35', '0.65', '0.8']
percentage = [1]
#percentage = ['0.8']
test = 1
for j in range(len(percentage)):
    # Where we save our checkpoints and graphs
    # exp_relative_dir = ['RNN Binary d0.9 s25 h64 b256 A1_partial_data_' + percentage[j], 'RNN Binary d0.9 s25 h64 b256 A2_partial_data_' + percentage[j],
    #                     'RNN Binary d0.9 s25 h64 b256 A3_partial_data_' + percentage[j], 'RNN Binary d0.9 s25 h64 b256 A4_partial_data_' + percentage[j]]
    exp_relative_dir = ['RNN Binary d0.9 s25 h64 b256 A1-4_all_data 0labeled 67 unlabeled 1000ep']
    #exp_relative_dir = ['RNN Binary d0.9 s25 h64 b256 Aniyama-dataport']

    # Which dataset we are targeting
    # dataset_dir = ['environment/time_series_repo/A1Benchmark', 'environment/time_series_repo/A2Benchmark',
    #                'environment/time_series_repo/A3Benchmark', 'environment/time_series_repo/A4Benchmark']
    dataset_dir = ['/Users/wuxiaodong/Downloads/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/']
    #dataset_dir = ['/Users/wuxiaodong/Dropbox/adaptive-anomalies/demo/csv/']
    #dataset_dir = ['/Users/wuxiaodong/Dropbox/adaptive-anomalies/Aniyama_groundtruth/dataport/']
    #dataset_dir = ['/home/lfwutong/sbsplusplus/datasets/Aniyama_groundtruth/dataport/']

    for i in range(len(dataset_dir)):
        env = EnvTimeSeriesfromRepo(dataset_dir[i])
        env.statefnc = RNNBinaryStateFuc
        env.rewardfnc = RNNBinaryRewardFuc
        env.timeseries_curser_init = n_steps
        env.datasetfix = DATAFIXED
        env.datasetidx = 0

        if test == 1:
            env.datasetrng = env.datasetsize
        else:
            env.datasetrng = np.int32(env.datasetsize * float(percentage[j]))

        experiment_dir = os.path.abspath("./exp/{}".format(exp_relative_dir[i]))

        tf.reset_default_graph()

        global_step = tf.Variable(0, name="global_step", trainable=False)

        qlearn_estimator = Q_Estimator_Nonlinear(scope="qlearn", summaries_dir=experiment_dir)
        target_estimator = Q_Estimator_Nonlinear(scope="target")

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        with sess.as_default():
            q_learning(env,
                       qlearn_estimator=qlearn_estimator,
                       target_estimator=target_estimator,
                       num_episodes=500,
                       num_epoches=10,
                       experiment_dir=experiment_dir,
                       replay_memory_size=500000,
                       replay_memory_init_size=50000,
                       update_target_estimator_every=10,
                       epsilon_start=1.0,
                       epsilon_end=0.1,
                       epsilon_decay_steps=500000,
                       discount_factor=0.9,
                       batch_size=256,
                       test=test)
            q_learning_validator(env, qlearn_estimator, 7)

