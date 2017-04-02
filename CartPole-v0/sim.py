import agent
import gym
from gym import wrappers
import numpy as np
import tensorflow as tf

def discountRewards(rewards):
    GAMMA = 0.99
    discounted_rewards = np.zeros_like(rewards)
    total = 0
    for i in reversed(xrange(0, rewards.size)):
        total = total * GAMMA + rewards[i]
        discounted_rewards[i] = total

    return discounted_rewards

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = wrappers.Monitor(env, 'exp/cartpole-experiment-1', force = True)

    ALPHA = 1e-2
    STATE_SIZE = 4
    ACTION_SIZE = 2
    HIDDEN_LAYER_SIZE = 3

    MAX_EPISODES = 5000
    UPDATE_FREQUENCY = 1
    CKPT_DIR = 'checkpoints/'

    tf.reset_default_graph()

    pilot = agent.agent(ALPHA, STATE_SIZE, ACTION_SIZE, HIDDEN_LAYER_SIZE)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        episode_num = 0
        rewards = []

        latest_ckpt = tf.train.latest_checkpoint(CKPT_DIR)
        if latest_ckpt != None:
            pilot.saver.restore(sess, latest_ckpt)
            print 'Restoring last checkpoint.'

        gradient_buffer = sess.run(tf.trainable_variables())
        for i, gradient in enumerate(gradient_buffer):
            gradient_buffer[i] = gradient * 0

        while episode_num < MAX_EPISODES:
            state = env.reset()

            episode_history = []
            running_reward = 0
            while True:
                if episode_num % 100 == 0:
                    env.render()

                action_dist = sess.run(pilot.output_layer, feed_dict = {pilot.input_layer: [state]})
                action = np.random.choice(action_dist[0], p = action_dist[0])
                action = np.argmax(action_dist == action)

                new_state, reward, done, _ = env.step(action)

                episode_history.append([state, action, reward, new_state])
                running_reward += reward
                state = new_state

                if done == True:
                    #Update the network
                    episode_history = np.array(episode_history)
                    states = np.vstack(episode_history[:, 0])
                    actions = episode_history[:, 1]
                    episode_history[:, 2] = discountRewards(episode_history[:, 2])

                    feed_dict = {
                        pilot.input_layer: np.vstack(episode_history[:, 0]),
                        pilot.action_holder: episode_history[:, 1],
                        pilot.reward_holder: episode_history[:, 2]
                    }

                    gradients = sess.run(pilot.gradients, feed_dict = feed_dict)
                    for i, gradient in enumerate(gradients):
                        gradient_buffer[i] += gradient

                    if episode_num % UPDATE_FREQUENCY == 0 and episode_num != 0:
                        feed_dict = dict(zip(pilot.gradient_holder, gradient_buffer))
                        sess.run(pilot.update_batch, feed_dict = feed_dict)

                        for i, gradient in enumerate(gradient_buffer):
                            gradient_buffer[i] = gradient * 0

                    rewards.append(running_reward)
                    break

            if episode_num % 100 == 0 and episode_num != 0:
                print "Episode #: %d Reward: %f" %(episode_num, np.mean(rewards[-100:]))

            if episode_num % 1000 == 0 and episode_num != 0:
                pilot.saver.save(sess, CKPT_DIR + 'data.ckpt')
                print "Checkpoint saved."

            episode_num += 1
