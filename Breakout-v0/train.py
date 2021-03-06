import gym
from gym import wrappers
import multiprocessing
from network import PolicyNetwork, ValueNetwork
import tensorflow as tf
import threading
from worker import Worker

NUM_ACTIONS = 6
CKPT_DIR = 'checkpoints/'
EXP_DIR = 'exp/'

if __name__ == '__main__':
    tf.reset_default_graph()

    with tf.device("/cpu:0"):
        with tf.variable_scope('global'):
            policy_network = PolicyNetwork(NUM_ACTIONS)
            value_network = ValueNetwork(reuse = True)

        saver = tf.train.Saver(keep_checkpoint_every_n_hours = 1, max_to_keep = 10)

        num_workers = multiprocessing.cpu_count()
        workers = []
        for i in range(num_workers):
            env = gym.make('SpaceInvaders-v0')
            env = wrappers.Monitor(env, EXP_DIR + 'SpaceInvaders-v0-exp-1')
            new_worker = Worker(
                'worker_' + str(i),
                env,
                policy_network,
                value_network,
                saver
            )
            workers.append(new_worker)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coordinator = tf.train.Coordinator()

        latest_ckpt = tf.train.latest_checkpoint(CKPT_DIR)
        if latest_ckpt != None:
            saver.restore(sess, latest_ckpt)
            print 'Restoring last checkpoint.'

        worker_threads = []
        for worker in workers:
            work = lambda: worker.run(sess, coordinator)
            thread = threading.Thread(target = work)
            thread.start()
            worker_threads.append(thread)

        coordinator.join(worker_threads)
