import gym
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
from time import sleep

# Look into cost fn constants

class AC_Network():
    def __init__(self, state_size, action_size, scope):
        with tf.variable_scope(scope):
            HIDDEN_LAYER_1_SIZE = 20
            HIDDEN_LAYER_2_SIZE = 20
            ALPHA = 1e-2
            W1 = 0.5
            W2 = 1.0

            self.input_layer = tf.placeholder(shape = [None, state_size], dtype = tf.float32)
            self.hidden_layer_1 = slim.fully_connected(self.input_layer, HIDDEN_LAYER_1_SIZE, biases_initializer = None, activation_fn = tf.nn.relu)
            # self.hidden_layer_2 = slim.fully_connected(self.hidden_layer_1, HIDDEN_LAYER_2_SIZE, biases_initializer = None, activation_fn = tf.nn.elu)

            self.policy_layer = slim.fully_connected(self.hidden_layer_1, ACTION_SIZE, biases_initializer = None, activation_fn = tf.tanh)
            self.value_layer = slim.fully_connected(self.hidden_layer_1, 1, biases_initializer = None, activation_fn = None)

            if scope != 'global':
                self.actions = tf.placeholder(shape = [None, action_size], dtype = tf.float32)
                self.target_value = tf.placeholder(shape = [None], dtype = tf.float32)
                self.advantages = tf.placeholder(shape = [None], dtype = tf.float32)

                self.value_function_loss = tf.reduce_sum(tf.square(self.target_value - tf.reshape(self.value_layer, [-1])))
                self.policy_loss = - tf.reduce_sum(tf.multiply(tf.transpose(tf.square(self.actions)), self.advantages))
                self.total_loss = W1 * self.value_function_loss + W2 * self.policy_loss

                train_vars_local = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.total_loss, train_vars_local)

                optimizer = tf.train.AdamOptimizer(learning_rate = ALPHA)
                train_vars_global = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_gradients = optimizer.apply_gradients(zip(self.gradients, train_vars_global))


class Worker():
    GAMMA = 0.99

    def __init__(self, number, env, state_size, action_size):
        self.name = 'worker_' + str(number)
        self.local_AC_Network = AC_Network(state_size, action_size, self.name)
        self.env = env

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        total = 0
        for i in reversed(xrange(0, rewards.size)):
            total = total * self.GAMMA + rewards[i]
            discounted_rewards[i] = total

        return discounted_rewards

    def train(self, sess, rollout, bootstrap_value):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        discounted_rewards = self.discount_rewards(rewards)
        next_states = rollout[:, 3]
        values = rollout[:, 4]
        value_plus = np.asarray(values.tolist() + [bootstrap_value])

        advantages = rewards + self.GAMMA * value_plus[1:] - value_plus[:-1]
        advantages = self.discount_rewards(advantages)

        feed_dict = {
            self.local_AC_Network.input_layer: np.vstack(states),
            self.local_AC_Network.actions: np.vstack(actions),
            self.local_AC_Network.target_value: discounted_rewards,
            self.local_AC_Network.advantages: advantages
        }

        loss, _ = sess.run([self.local_AC_Network.total_loss, self.local_AC_Network.apply_gradients], feed_dict = feed_dict)

        print "Loss: %f" %(loss)

    def work(self, sess, coordinator):
        MAX_EPISODES = 10000

        print "Starting " + self.name

        with sess.as_default() and sess.graph.as_default():
            episode_count = 0
            while not coordinator.should_stop() and episode_count < MAX_EPISODES:
                episode_history = []
                episode_reward = 0
                state = self.env.reset()
                done = False

                while not done:
                    actions, value = sess.run([
                        self.local_AC_Network.policy_layer,
                        self.local_AC_Network.value_layer
                    ],
                    feed_dict = {
                        self.local_AC_Network.input_layer: [state]
                    })

                    actions = np.squeeze(actions)
                    next_state, reward, done, _ = env.step(actions)
                    episode_history.append([state, actions, reward, next_state, value[0, 0]])
                    episode_reward += reward
                    state = next_state

                if len(episode_history) > 0:
                    self.train(sess, episode_history, 0.0)

                print "Episode #%d: Reward: %f" %(episode_count, episode_reward)

                episode_count += 1


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2')

    STATE_SIZE = 24
    ACTION_SIZE = 4

    MAX_EPISODES = 10000
    UPDATE_FREQUENCY = 5

    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        master_network = AC_Network(STATE_SIZE, ACTION_SIZE, 'global')
        num_workers = multiprocessing.cpu_count()

        workers = []
        for i in range(num_workers):
            new_worker = Worker(i, env, STATE_SIZE, ACTION_SIZE)
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
