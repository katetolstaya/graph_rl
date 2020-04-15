import numpy as np
import gym
import gym_flock
import tensorflow as tf
from progress.bar import Bar


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
