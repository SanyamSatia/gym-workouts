import gym
from atari_environment import AtariEnvironment
import multiprocessing
import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
from time import sleep

GAMMA = 0.9

def make_copy_params_op(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    return op_holder

def discount(x):
    return scipy.signal.lfilter([1], [1, -GAMMA], x[::-1], axis=0)[::-1]

class AC_Network():
    def __init__(self, action_size, scope, optimizer):
        with tf.variable_scope(scope):
            self.num_actions = action_size

            self.input_layer = tf.placeholder(shape = [None, 4, 84, 84], dtype = tf.float32)
            normalized_input = tf.div(tf.to_float(self.input_layer), 255.0)
            convolution_layer_1 = tf.contrib.layers.conv2d(normalized_input, 16, 8, 4, activation_fn = tf.nn.relu)
            convolution_layer_2 = tf.contrib.layers.conv2d(convolution_layer_1, 32, 4, 2, activation_fn = tf.nn.relu)
            fully_connected_layer = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(convolution_layer_2), 256)

            self.policy_layer = tf.contrib.layers.fully_connected(fully_connected_layer, self.num_actions, activation_fn = tf.nn.softmax)
            self.value_layer = tf.contrib.layers.fully_connected(fully_connected_layer, 1, activation_fn = None)

            if scope != 'global':
                self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, action_size, dtype=tf.float32)
                self.target_values = tf.placeholder(shape = [None], dtype = tf.float32)
                self.advantages = tf.placeholder(shape = [None], dtype = tf.float32)

                self.chosen_actions = tf.reduce_sum(self.policy_layer * self.actions_onehot, [1])

                self.value_function_loss = tf.reduce_sum(tf.squared_difference(self.target_values, tf.reshape(self.value_layer, [-1])))
                self.policy_loss = - tf.reduce_sum(tf.log(self.chosen_actions) * self.advantages)
                self.entropy = - tf.reduce_sum(self.policy_layer * tf.log(self.policy_layer))

                self.total_loss = 0.5 * self.value_function_loss + self.policy_loss - 0.01 * self.entropy

                train_vars_local = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.total_loss, train_vars_local)
                self.var_norm = tf.global_norm(train_vars_local)
                grads, self.grad_norm = tf.clip_by_global_norm(self.gradients, 40.0)

                train_vars_global = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_gradients = optimizer.apply_gradients(zip(grads, train_vars_global))


class Worker():
    MAX_EPISODE_COUNT = 1000

    def __init__(self, number, env, action_size, optimizer, global_episode_count):
        RESIZED_WIDTH = 84
        RESIZED_HEIGHT = 84
        AGENT_HISTORY_LENGTH = 4
        LOG_DIR = 'logs/'

        self.name = 'worker_' + str(number)
        self.optimizer = optimizer
        self.local_AC_Network = AC_Network(action_size, self.name, self.optimizer)
        self.env = AtariEnvironment(gym_env = env, resized_width = RESIZED_WIDTH, resized_height = RESIZED_HEIGHT, agent_history_length = AGENT_HISTORY_LENGTH)
        self.num_actions = action_size
        self.global_episode_count = global_episode_count
        self.increment_global_episode_count = self.global_episode_count.assign_add(1)
        self.summary_writer = tf.summary.FileWriter(LOG_DIR + self.name)

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        self.update_local_network = make_copy_params_op('global', self.name)

    def train(self, sess, rollout, bootstrap_value):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_states = rollout[:, 3]
        values = rollout[:, 4]

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + GAMMA * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages)

        feed_dict = {
            self.local_AC_Network.input_layer: np.array(states.tolist()),
            self.local_AC_Network.actions: actions,
            self.local_AC_Network.target_values: discounted_rewards,
            self.local_AC_Network.advantages: advantages
        }

        vf_loss, pi_loss, entropy, loss, grad_norm, var_norm, _ = sess.run([
            self.local_AC_Network.value_function_loss,
            self.local_AC_Network.policy_loss,
            self.local_AC_Network.entropy,
            self.local_AC_Network.total_loss,
            self.local_AC_Network.grad_norm,
            self.local_AC_Network.var_norm,
            self.local_AC_Network.apply_gradients
        ], feed_dict = feed_dict)

        return vf_loss, pi_loss, entropy, loss, grad_norm, var_norm

    def work(self, sess, coordinator):
        MAX_EPISODES = 1000
        EPISODE_BUFFER_SIZE = 30
        LOG_FREQUENCY = 2

        print "Starting " + self.name

        episode_count = sess.run(self.global_episode_count)

        with sess.as_default() and sess.graph.as_default():
            while not coordinator.should_stop():
                sess.run(self.update_local_network)
                episode_history = []
                episode_values = []
                episode_reward = 0
                state = self.env.get_initial_state()
                done = False

                while not done:
                    action_dist, value = sess.run([
                        self.local_AC_Network.policy_layer,
                        self.local_AC_Network.value_layer
                    ],
                    feed_dict = {
                        self.local_AC_Network.input_layer: [state]
                    })

                    action_dist = np.squeeze(action_dist)
                    action = np.random.choice(range(len(action_dist)), p = action_dist)
                    value = np.squeeze(value)
                    next_state, reward, done, _ = self.env.step(action)
                    episode_history.append([state, action, reward, next_state, value])
                    episode_values.append(value)
                    episode_reward += reward
                    state = next_state

                    if len(episode_history) >= EPISODE_BUFFER_SIZE and not done:
                        value = sess.run(self.local_AC_Network.value_layer,
                            feed_dict = {
                                self.local_AC_Network.input_layer: [state]
                            }
                        )
                        value = np.squeeze(value)
                        vf_loss, pi_loss, entropy, loss, grad_norm, var_norm = self.train(sess, episode_history, value)
                        episode_history = []

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(len(episode_values))
                self.episode_mean_values.append(np.mean(episode_values))

                if len(episode_history) > 0:
                    vf_loss, pi_loss, entropy, loss, grad_norm, var_norm = self.train(sess, episode_history, 0.0)

                if episode_count % LOG_FREQUENCY == 0 and episode_count != 0:
                    mean_reward = np.mean(self.episode_rewards[-LOG_FREQUENCY:])
                    mean_length = np.mean(self.episode_lengths[-LOG_FREQUENCY:])
                    mean_value = np.mean(self.episode_mean_values[-LOG_FREQUENCY:])

                    summary = tf.Summary()
                    summary.value.add(tag = 'Perf/Reward', simple_value = float(mean_reward))
                    summary.value.add(tag = 'Perf/Length', simple_value = float(mean_length))
                    summary.value.add(tag = 'Perf/Value', simple_value = float(mean_value))
                    summary.value.add(tag = 'Losses/Value Loss', simple_value = float(vf_loss))
                    summary.value.add(tag = 'Losses/Policy Loss', simple_value = float(pi_loss))
                    summary.value.add(tag = 'Losses/Entropy', simple_value = float(entropy))
                    summary.value.add(tag = 'Losses/Total', simple_value = float(loss))
                    summary.value.add(tag = 'Losses/Grad Norm', simple_value = float(grad_norm))
                    summary.value.add(tag = 'Losses/Var Norm', simple_value = float(var_norm))
                    self.summary_writer.add_summary(summary, episode_count)
                    self.summary_writer.flush()

                if self.name == "worker_0":

                    print "Episode #%d:\nvf_loss: %f pi_loss: %f entropy: %f\nLoss: %f Reward: %f" %(episode_count, vf_loss, pi_loss, entropy, loss, episode_reward)
                    sess.run(self.increment_global_episode_count)

                episode_count += 1


if __name__ == '__main__':
    ACTION_SIZE = 3
    ALPHA = 1e-3

    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        global_episode_count = tf.Variable(0, dtype = tf.int32, trainable = False)
        optimizer = tf.train.AdamOptimizer(learning_rate = ALPHA)
        master_network = AC_Network(ACTION_SIZE, 'global', optimizer)
        num_workers = 2 #multiprocessing.cpu_count()

        workers = []
        for i in range(num_workers):
            env = gym.make('Pong-v0')
            new_worker = Worker(i, env, ACTION_SIZE, optimizer, global_episode_count)
            workers.append(new_worker)

    with tf.Session() as sess:
        coordinator = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            work = lambda: worker.work(sess, coordinator)
            thread = threading.Thread(target = (work))
            thread.start()
            sleep(0.5)
            worker_threads.append(thread)

        coordinator.join(worker_threads)
