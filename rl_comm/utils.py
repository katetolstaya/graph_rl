import numpy as np
import random
import gym
import gym_flock
import tensorflow as tf
from graph_nets.blocks import unsorted_segment_max_or_zero
from progress.bar import Bar


def segment_logsumexp(data, segments_ids, num_segments, name=None):
    """ Compute logsumexp over segments via logsumexp trick """
    # reference: https://github.com/becxer/rejection-qa/blob/master/docqa/nn/ops.py#L12
    segment_maxes = unsorted_segment_max_or_zero(data, segments_ids, num_segments, name)
    data -= tf.gather(segment_maxes, segments_ids, axis=0)
    data_log = tf.log(tf.math.unsorted_segment_sum(tf.exp(data), segments_ids, num_segments))
    # relu clips -inf values for segments with no data
    return tf.nn.relu(data_log) + segment_maxes


def ckpt_file(ckpt_dir, ckpt_idx):
    return ckpt_dir / 'ckpt_{:03}.pkl'.format(ckpt_idx)


def print_key_if_true(dictionary, key):
    """
    Print each key string whose value in dictionary is True.
    """
    if dictionary[key]:
        return key + ', '
    return ''


def eval_env(env, model, n_episodes, render_mode='none'):
    """
    Evaluate a model against an environment over N games.
    """
    results = {
        'reward': np.zeros(n_episodes),
    }
    with Bar('Eval', max=n_episodes) as bar:
        for k in range(n_episodes):
            done = False
            obs = env.reset()
            ep_reward = 0
            # Run one game.
            while not done:
                action, states = model.predict(obs, deterministic=True)  # TODO need to reformat here?
                obs, r, done, _ = env.step(action)
                ep_reward += r
                # env.render(mode=render_mode)
            # Record results.
            results['reward'][k] = ep_reward
            bar.next()
    return results


def callback(locals_, globals_, test_env, interval, n_episodes=50):
    self_ = locals_['self']

    # Periodically run extra test evaluation.
    if not hasattr(self_, 'next_test_eval'):
        self_.next_test_eval = 0
    if self_.num_timesteps >= self_.next_test_eval:
        print('\nTesting...')
        results = eval_env(test_env, self_, n_episodes, render_mode='none')
        print('reward,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['reward']),
                                                                    np.std(results['reward'])))
        print('')
        score = np.mean(results['reward'])
        summary = tf.Summary(value=[tf.Summary.Value(tag='reward', simple_value=score)])
        locals_['writer'].add_summary(summary, self_.num_timesteps)
        self_.next_test_eval += interval
    return True


class ReplayBuffer(object):
    """
    Stores training samples for RL algorithms, represented as tuples of (S, A, R, S).
    """

    def __init__(self, max_size=1000):
        """
        Initialize the replay buffer object. Once the buffer is full, remove the oldest sample.
        :param max_size: maximum size of the buffer.
        """
        self.buffer = []
        self.max_size = max_size
        self.curr_size = 0
        self.position = 0

    def insert(self, sample):
        """
        Insert sample into buffer.
        :param sample: The (S,A,R,S) tuple.
        :return: None
        """
        if self.curr_size < self.max_size:
            self.buffer.append(None)
            self.curr_size = self.curr_size + 1

        self.buffer[self.position] = sample
        self.position = (self.position + 1) % self.max_size

    def sample(self, num_samples):
        """
        Sample a number of transitions from the replay buffer.
        :param num_samples: Number of transitions to sample.
        :return: The set of sampled transitions.
        """
        return random.sample(self.buffer, num_samples)

    def clear(self):
        """
        Clears the current buffer.
        :return: None
        """
        self.buffer = []
        self.curr_size = 0
        self.position = 0