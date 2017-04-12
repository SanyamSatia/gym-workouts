# Based on https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/a3c/estimators.py

import tensorflow as tf

ENTROPY_WEIGHT = 1e-2
LEARNING_RATE = 1e-4
DECAY_RATE = 0.99
MOMENTUM = 0.0
EPSILON = 1e-6

def build_shared_network(inputs):
    convolution_layer_1 = tf.contrib.layers.conv2d(inputs, 16, 8, 4, activation_fn = tf.nn.relu)
    convolution_layer_2 = tf.contrib.layers.conv2d(convolution_layer_1, 32, 4, 2, activation_fn = tf.nn.relu)

    fully_connected_layer = tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(convolution_layer_2), 256)

    return fully_connected_layer

class PolicyNetwork():
    def __init__(self, num_actions):
        self.num_actions = num_actions

        self.states = tf.placeholder(shape = [None, 4, 84, 84], dtype = tf.uint8)
        self.targets = tf.placeholder(shape = [None], dtype = tf.float32)
        self.actions = tf.placeholder(shape = [None], dtype = tf.int32)

        inputs = tf.to_float(self.states) / 255.0

        with tf.variable_scope("shared"):
            fully_connected_layer = build_shared_network(inputs)

        with tf.variable_scope("policy_network"):
            self.action_dist = tf.contrib.layers.fully_connected(fully_connected_layer, self.num_actions, activation_fn = tf.nn.softmax)

            self.entropy = - tf.reduce_sum(self.action_dist * tf.log(self.action_dist), 1)

            gather_indices = tf.range(tf.shape(self.states)[0]) * tf.shape(self.action_dist)[1] + self.actions
            self.chosen_action_prob = tf.gather(tf.reshape(self.action_dist, [-1]), gather_indices)

            self.loss = tf.log(self.chosen_action_prob) * self.targets + ENTROPY_WEIGHT * self.entropy
            self.loss = tf.reduce_sum(- self.loss)

            self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, DECAY_RATE, MOMENTUM, EPSILON)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
            self.train = self.optimizer.apply_gradients(self.grads_and_vars, global_step = tf.contrib.framework.get_global_step())

class ValueNetwork():
    def __init__(self):
        self.states = tf.placeholder(shape = [None, 4, 84, 84], dtype = tf.uint8)
        self.targets = tf.placeholder(shape = [None], dtype = tf.float32)

        inputs = tf.to_float(self.states) / 255.0

        with tf.variable_scope("shared"):
            fully_connected_layer = build_shared_network(inputs)

        with tf.variable_scope("value_network"):
            self.logits = tf.contrib.layers.fully_connected(fully_connected_layer, 1, activation_fn = None)
            self.logits = tf.squeeze(self.logits, squeeze_dims = [1])

            self.loss = tf.reduce_sum(tf.squared_difference(self.logits, self.targets))

            self.optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, DECAY_RATE, MOMENTUM, EPSILON)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
            self.train = self.optimizer.apply_gradients(self.grads_and_vars, global_step = tf.contrib.framework.get_global_step())
