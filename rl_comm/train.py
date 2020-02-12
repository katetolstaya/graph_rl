import glob
import numpy as np
import functools
from pathlib import Path
import gym
import gym_flock
import tensorflow as tf
from progress.bar import Bar

from rl_comm.ppo2 import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.gail import ExpertDataset

import rl_comm.gnn_fwd as gnn_fwd


def ckpt_file(ckpt_dir, ckpt_idx):
    return ckpt_dir / 'ckpt_{:03}.pkl'.format(ckpt_idx)


def print_key_if_true(dictionary, key):
    """
    Print each key string whose value in dictionary is True.
    """
    if dictionary[key] == True:
        return key + ', '
    return ''


def eval_env(env, model, N, render_mode='none'):
    """
    Evaluate a model against an environment over N games.
    """
    results = {
        'reward': np.zeros(N),
    }
    with Bar('Eval', max=N) as bar:
        for k in range(N):
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


def callback(locals_, globals_, test_env):
    self_ = locals_['self']

    # Periodically run extra test evaluation.
    if not hasattr(self_, 'next_test_eval'):
        self_.next_test_eval = 0
    if self_.num_timesteps >= self_.next_test_eval:
        print('\nTesting...')
        results = eval_env(test_env, self_, 50, render_mode='none')
        print('reward,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['reward']), np.std(results['reward'])))
        print('')
        score = np.mean(results['reward'])
        summary = tf.Summary(value=[tf.Summary.Value(tag='reward', simple_value=score)])
        locals_['writer'].add_summary(summary, self_.num_timesteps)
        self_.next_test_eval += 50000
    return True


def train_helper(env_param, test_env_param, train_param, policy_fn, policy_param, directory):
    save_dir = Path(directory)
    tb_dir = save_dir / 'tb'
    ckpt_dir = save_dir / 'ckpt'
    for d in [save_dir, tb_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    env_name = "MappingRad-v0"

    def make_env():
        keys = ['nodes', 'edges', 'senders', 'receivers']
        env = gym.make(env_name)
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=keys)
        return env

    # env = VecNormalize(SubprocVecEnv([make_env]*train_param['n_env']), norm_obs=False, norm_reward=True)

    env = SubprocVecEnv([make_env]*train_param['n_env'])
    test_env = SubprocVecEnv([make_env])

    # Find latest checkpoint index.
    # ckpt_list = sorted(glob.glob(str(ckpt_dir) + '/*.pkl'))
    # if len(ckpt_list) == 0:
    #     ckpt_idx = None
    # else:
    #     ckpt_idx = int(ckpt_list[-1][-7:-4])

    ckpt_idx = None

    # Load or create model.
    if ckpt_idx is not None:
        print('\nLoading model {}.\n'.format(ckpt_file(ckpt_dir, ckpt_idx).name))
        model = PPO2.load(str(ckpt_file(ckpt_dir, ckpt_idx)), env, tensorboard_log=str(tb_dir))
        ckpt_idx += 1
    else:
        print('\nCreating new model.\n')
        model = PPO2(
            policy=policy_fn,
            policy_kwargs=policy_param,
            env=env,
            # learning_rate=1e-6,
            learning_rate=5e-6,
            cliprange=1.0,
            n_steps=train_param['n_steps'],
            ent_coef=0.01,
            vf_coef=0.5,
            verbose=1,
            tensorboard_log=str(tb_dir),
            full_tensorboard_log=False)
        ckpt_idx = 0

        # model_name = 'ckpt_026.pkl'
        #
        # # load the dictionary of parameters from file
        # _, params = BaseRLModel._load_from_file(model_name)
        #
        # # update new model's parameters
        # model.load_parameters(params)

    dataset = ExpertDataset(expert_path='data/expert_multi.npz',
                            traj_limitation=-1, batch_size=16)
    model.pretrain(dataset, n_epochs=1000, learning_rate=5e-6)

    # dataset = ExpertDataset(expert_path='data/expert_rad2.npz',
    #                         traj_limitation=-1, batch_size=16)
    # model.pretrain(dataset, n_epochs=5000, learning_rate=1e-6)
    # model.pretrain(dataset, n_epochs=200, learning_rate=1e-5)
    # model.pretrain(dataset, n_epochs=1000, learning_rate=5e-6)

    # Training loop.
    print('\nBegin training.\n')
    while model.num_timesteps <= train_param['total_timesteps']:
        print('\nLearning...\n')
        model.learn(
            total_timesteps=train_param['checkpoint_timesteps'],
            log_interval=500,
            reset_num_timesteps=False,
            callback=functools.partial(callback, test_env=test_env))

        print('\nSaving model {}.\n'.format(ckpt_file(ckpt_dir, ckpt_idx).name))
        model.save(str(ckpt_file(ckpt_dir, ckpt_idx)))
        ckpt_idx += 1

    print('Finished.')


if __name__ == '__main__':

    import copy

    jobs = []  # string name, policy class, policy_kwargs

    j = {}
    j['policy'] = gnn_fwd.GnnFwd
    # j['policy'] = MlpPolicy
    j['policy_param'] = {'num_processing_steps': 5}
    # j['name'] = j['policy'].policy_param_string(j['policy_param'])
    j['name'] = 'vrp'
    jobs.append(j)

    env_param = {}
    test_env_param = copy.deepcopy(env_param)

    train_param = {
        'n_env': 16,
        'n_steps': 32,
        'checkpoint_timesteps': 100000,
        'total_timesteps': 50000000
    }

    root = Path('models/' + j['name'])

    for j in jobs:
        directory = root / j['name']  # env_param_string(env_param) / train_param_string(train_param) / j['name']

        train_helper(
            env_param=env_param,
            test_env_param=test_env_param,
            train_param=train_param,
            policy_fn=j['policy'],
            policy_param=j['policy_param'],
            directory=directory)
