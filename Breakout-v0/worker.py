# Based on https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/a3c/estimators.py

from atari_environment import AtariEnvironment
from network import PolicyNetwork, ValueNetwork
import numpy as np
import tensorflow as tf

RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84
AGENT_HISTORY_LENGTH = 4
DISCOUNT_FACTOR = 0.99
MAX_GLOBAL_EPISODES = 1000
UPDATE_FREQUENCY = 30
RENDER_FREQUENCY = 25
CKPT_DIR = 'checkpoints/'
CKPT_FREQUENCY = 100

def get_copy_params_op(global_vars, local_vars):
    op_holder = []
    for global_var, local_var in zip(global_vars, local_vars):
        op_holder.append(local_var.assign(global_var))
    return op_holder

def get_train_op(local_network, global_network):
    local_grads, _ = zip(*local_network.grads_and_vars)
    local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
    _, global_vars = zip(*global_network.grads_and_vars)
    local_grads_global_vars = list(zip(local_grads, global_vars))

    return global_network.optimizer.apply_gradients(local_grads_global_vars, global_step = tf.contrib.framework.get_global_step())

class Worker():
    def __init__(self, name, env, policy_network, value_network, saver):
        self.name = name
        self.global_policy_network = policy_network
        self.global_value_network = value_network
        self.env = env
        self.atari_env = AtariEnvironment(gym_env = self.env, resized_width = RESIZED_WIDTH, resized_height = RESIZED_HEIGHT, agent_history_length = AGENT_HISTORY_LENGTH)
        self.saver = saver

        with tf.variable_scope(self.name):
            self.policy_network = PolicyNetwork(self.global_policy_network.num_actions)
            self.value_network = ValueNetwork(reuse = True)

        self.copy_params_op = get_copy_params_op(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global'),
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        )

        self.policy_network_train_op = get_train_op(self.policy_network, self.global_policy_network)
        self.value_network_train_op = get_train_op(self.value_network, self.global_value_network)

    def run(self, sess, coordinator):
        with sess.as_default() and sess.graph.as_default():
            episode_count = 0

            while not coordinator.should_stop():
                episode_history = []
                done = False
                episode_length = 0
                episode_reward = 0
                state = self.atari_env.get_initial_state()

                while not done:
                    sess.run(self.copy_params_op)

                    # if episode_count % RENDER_FREQUENCY == 0 and self.name == 'worker_0':
                    #     self.atari_env.render()

                    action_dist = sess.run(self.policy_network.action_dist, feed_dict = {self.policy_network.states: [state]})
                    action_dist = np.squeeze(action_dist)
                    action = np.random.choice(range(len(action_dist)), p = action_dist)
                    next_state, reward, done, _ = self.atari_env.step(action)
                    episode_history.append([state, action, reward, next_state])

                    state = next_state
                    episode_length += 1
                    episode_reward += reward

                    if done or (episode_length != 0 and episode_length % UPDATE_FREQUENCY == 0):
                        pi_loss, vf_loss = self.update(sess, episode_history, done)
                        episode_history = []
                        # print "%s: Episode #: %d pi_loss: %f vf_loss: %f" %(self.name, episode_count, pi_loss, vf_loss)

                    # if self.global_counter >= MAX_GLOBAL_EPISODES:
                    #     coordinator.request_stop()

                print "%s: Episode #: %d Reward: %d" %(self.name, episode_count, episode_reward)

                if self.name == 'worker_0' and episode_count % CKPT_FREQUENCY == 0 and episode_count != 0:
                    self.saver.save(sess, CKPT_DIR + 'data.ckpt')
                    print "Checkpoint saved."

                episode_count += 1

    def update(self, sess, episode_history, done):
        reward = 0.0
        if not done:
            reward = np.squeeze(sess.run(self.value_network.logits, feed_dict = {self.value_network.states: [episode_history[-1][3]]}))

        states = []
        actions = []
        policy_targets = []
        value_targets = []

        for i in reversed(range(len(episode_history))):
            states.append(episode_history[i][0])
            actions.append(episode_history[i][1])
            reward = episode_history[i][2] + DISCOUNT_FACTOR * reward
            value = np.squeeze(sess.run(self.value_network.logits, feed_dict = {self.value_network.states: [episode_history[i][0]]}))
            policy_target = reward - value
            policy_targets.append(policy_target)
            value_targets.append(reward)

        feed_dict = {
            self.policy_network.states: np.array(states),
            self.policy_network.targets: policy_targets,
            self.policy_network.actions: actions,
            self.value_network.states: np.array(states),
            self.value_network.targets: value_targets
        }

        pi_loss, vf_loss, _, _ = sess.run([
            self.policy_network.loss,
            self.value_network.loss,
            self.policy_network_train_op,
            self.value_network_train_op
        ], feed_dict = feed_dict)

        return pi_loss, vf_loss
