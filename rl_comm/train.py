import glob
import numpy as np
import functools
from pathlib import Path

import gym
# env_dict = gym.envs.registration.registry.env_specs.copy()
# for env in env_dict:
#     print('Remove {} from registry'.format(env))
#     del gym.envs.registration.registry.env_specs[env]

import gym_flock
import tensorflow as tf
from progress.bar import Bar

from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines import PPO2

import rl_comm.gnn_fwd as gnn_fwd


# def train_param_string(p):
#     """Return identifier string for A2C training parameter dict."""
#     return 'ne_{ne}_ns_{ns}'.format(
#         ne=p['n_env'],
#         ns=p['n_steps'])
#

# def env_param_string(p):
#     """Return identifier string for PDefenseEnv environment parameter dict."""
#     comm = {'clique': 'clq', 'circulant': 'cir', 'range': 'rng'}[p['comm_adj_type']]
#     if p['comm_adj_type'] == 'range':
#         comm = comm + '_{}'.format(p['comm_adj_r'])
#     return 'na_{na}_rc_{rc}_fov_{fv}_{comm}'.format(
#         na=p['n_max_agents'],
#         rc=p['r_capture'],
#         fv=p['fov'],
#         comm=comm)


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
        # 'score': np.zeros(N),
        # 'lgr_score': np.zeros(N),
        # 'initial_lgr_score': np.zeros(N)
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
            # results['score'][k] = info['score']
            # results['lgr_score'][k] = info['lgr_score']
            # results['initial_lgr_score'][k] = info['initial_lgr_score']
            bar.next()
    return results


def callback(locals_, globals_, test_env):
    self_ = locals_['self']

    # Add extra tensors to normal logging.
    # Can't do it like this, because the existing summary is not invoked during the policy run.
    # if not hasattr(self_, 'is_tb_set'):
    #     print('Only once.')
    #     with self_.graph.as_default():
    #         print('Once.')
    #         # print(self_.act_model.msg_enc_g.edges.shape)
    #         tf.summary.histogram('msg_enc', self_.act_model.msg_enc_g.edges)
    #         # tf.summary.histogram('msg_bin', self_.act_model.msg_bin_g.edges)
    #         self_.summary = tf.summary.merge_all()
    #     self_.is_tb_set = True

    # Periodically run extra test evaluation.
    if not hasattr(self_, 'next_test_eval'):
        self_.next_test_eval = 0
    if self_.num_timesteps >= self_.next_test_eval:
        print('\nTesting...')
        results = eval_env(test_env, self_, 50, render_mode='none')
        print('reward,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['reward']), np.std(results['reward'])))
        # print('init_lgr_score, mean = {:.1f}, std = {:.1f}'.format(np.mean(results['initial_lgr_score']),
        #                                                            np.std(results['initial_lgr_score'])))
        # print('steps,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['steps']), np.std(results['steps'])))
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
    ckpt_list = sorted(glob.glob(str(ckpt_dir) + '/*.pkl'))
    if len(ckpt_list) == 0:
        ckpt_idx = None
    else:
        ckpt_idx = int(ckpt_list[-1][-7:-4])

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
            learning_rate=1e-6,
            cliprange=1.0,
            n_steps=train_param['n_steps'],
            ent_coef=0.0,  # 0.01,
            vf_coef=0.5,
            verbose=1,
            tensorboard_log=str(tb_dir),
            full_tensorboard_log=False)
        ckpt_idx = 0

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

    # Miniature ICRA 2018 with msg_size = 8 and global vf head.
    # j = {}
    # j['policy'] = gnn_fwd.GnnFwd
    # j['policy_param'] = {
    #     'input_feat_layers': (64, 64),
    #     'feat_agg_layers': (),
    #     'msg_enc_layers': (64, 64),
    #     'msg_size': 8,
    #     'msg_dec_layers': (64, 64),
    #     'msg_agg_layers': (64, 64),
    #     'pi_head_layers': (),
    #     'vf_local_head_layers': (),
    #     'vf_global_head_layers': (64,)}
    # j['name'] = j['policy'].policy_param_string(j['policy_param'])
    # jobs.append(j)

    # Miniature ICRA 2018 with msg_size = 0 and global vf head.
    j = {}
    j['policy'] = gnn_fwd.GnnFwd
    # j['policy_param'] = {
    #     'input_feat_layers': (64, 64),
    #     'feat_agg_layers': (),
    #     'msg_enc_layers': (64, 64),
    #     'msg_size': 0,
    #     'msg_dec_layers': (64, 64),
    #     'msg_agg_layers': (64, 64),
    #     'pi_head_layers': (),
    #     'vf_local_head_layers': (),
    #     'vf_global_head_layers': (64,)}
    # j['name'] = j['policy'].policy_param_string(j['policy_param'])
    j['name'] = '2019-09-22'
    jobs.append(j)

    env_param = {}
    test_env_param = copy.deepcopy(env_param)

    train_param = {
        'n_env': 16,
        'n_steps': 32,
        'checkpoint_timesteps': 1000000,
        'total_timesteps': 50000000
    }

    root = Path('models/2019-09-13')

    for j in jobs:
        directory = root / j['name']  # env_param_string(env_param) / train_param_string(train_param) / j['name']

        train_helper(
            env_param=env_param,
            test_env_param=test_env_param,
            train_param=train_param,
            policy_fn=j['policy'],
            policy_param=None,  # j['policy_param'],
            directory=directory)
