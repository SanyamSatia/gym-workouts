import agent
import gym
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    ALPHA = 1e-2
    STATE_SIZE = 8
    ACTION_SIZE = 4
    HIDDEN_LAYER_SIZE = 8
    MAX_EPISODES = 10000
    CKPT_SOLVED_DIR = 'checkpoints/solved/'

    pilot = agent.agent(ALPHA, STATE_SIZE, ACTION_SIZE, HIDDEN_LAYER_SIZE)

    latest_ckpt = tf.train.latest_checkpoint(CKPT_SOLVED_DIR)
    if latest_ckpt == None:
        print 'No checkpoints found. Exiting.'
        exit()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, latest_ckpt)
        print 'Restoring last checkpoint.'

        for i in range(MAX_EPISODES):
            state = env.reset()
            reward = 0
            while True:
                env.render()
                action_dist = sess.run(pilot.output_layer, feed_dict = {pilot.input_layer: [state]})
                action = np.random.choice(action_dist[0], p = action_dist[0])
                action = np.argmax(action_dist == action)
                state, cur_reward, done, _ = env.step(action)
                reward += cur_reward

                if done == True:
                    print 'Episode #%d: Reward: %f' %(i, reward)
                    break
