import tensorflow as tf
import tensorflow.contrib.slim as slim

class agent():
    def __init__(self, learning_rate, state_size, action_size, hidden_layer_size):

        self.input_layer = tf.placeholder(shape = [None, state_size], dtype = tf.float32)
        self.hidden_layer_1 = slim.fully_connected(self.input_layer, hidden_layer_size, biases_initializer = None, activation_fn = tf.nn.relu)
        self.hidden_layer_2 = slim.fully_connected(self.hidden_layer_1, hidden_layer_size, biases_initializer = None, activation_fn = tf.nn.relu)
        self.output_layer = slim.fully_connected(self.hidden_layer_2, action_size, biases_initializer = None, activation_fn = tf.nn.softmax)

        self.best_action = tf.argmax(self.output_layer, 1)

        self.reward_holder = tf.placeholder(shape = [None], dtype = tf.float32)
        self.action_holder = tf.placeholder(shape = [None], dtype = tf.int32)

        self.indices = tf.range(0, tf.shape(self.output_layer)[0]) * tf.shape(self.output_layer)[1] + self.action_holder
        self.outputs = tf.gather(tf.reshape(self.output_layer, [-1]), self.indices)
        self.cost = -tf.reduce_mean(tf.log(self.outputs) * self.reward_holder)

        train_vars = tf.trainable_variables()
        self.gradient_holder = []
        for i, var in enumerate(train_vars):
            placeholder = tf.placeholder(dtype = tf.float32, name = str(i) + '_holder')
            self.gradient_holder.append(placeholder)

        self.gradients = tf.gradients(self.cost, train_vars)

        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holder, train_vars))

        self.saver = tf.train.Saver()
