import os.path
import numpy as np
import functools

import tensorflow as tf

from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

from gym_pdefense.envs.pdefense_env_lgr import PDefenseEnv
import gnn_fwd

def train_param_string(p):
    """Return identifier string for A2C training parameter dict."""
    return 'ne_{ne}_ns_{ns}'.format(
        ne=p['n_env'],
        ns=p['n_steps'])

def env_param_string(p):
    """Return identifier string for PDefenseEnv environment parameter dict."""
    comm_prefix = {'clique':'clq', 'circulant':'cir', 'range':'rng'}[p['comm_adj_type']]
    if p['comm_adj_type'] == 'range':
        postfix = '{}'.format(p['comm_adj_r'])
    else:
        comm_postfix = ''
    return 'na_{na}_rc_{rc}_{comm}'.format(
        na=p['n_max_agents'],
        rc=p['r_capture'],
        comm=comm_prefix+comm_postfix)

def print_key_if_true(dictionary, key):
    """
    Print each key string whose value in dictionary is True.
    """
    if dictionary[key] == True:
        return key + ', '
    return ''

def eval_pdefense_env(env, model, N, render_mode='none'):
    """
    Evaluate a model against an environment over N games.
    """
    print()
    print('testing')

    results = {
        'steps': np.zeros(N),
        'score': np.zeros(N),
        'lgr_score': np.zeros(N),
        'initial_lgr_score': np.zeros(N)
    }

    for k in range(N):
        print('.', end='', flush=True)
        done = False
        obs = env.reset()
        # Run one game.
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            env.render(mode=render_mode) # pick from ['none', human', 'ffmpeg']

        # # Display results.
        # cause = ''.join([print_key_if_true(info, key) for key in
        #     ['all_agents_dead', 'all_targets_dead', 'lgr_score_increased', 'no_more_rewards']])
        # print('{:>2} {}={}+{} {}'.format(
        #     info['steps'],
        #     info['initial_lgr_score'],
        #     info['score'],
        #     info['lgr_score'],
        #     cause))

        # Record results.
        results['steps'][k] = info['steps']
        results['score'][k] = info['score']
        results['lgr_score'][k] = info['lgr_score']
        results['initial_lgr_score'][k] = info['initial_lgr_score']

    print('')
    print('score,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['score']), np.std(results['score'])))
    print('init_lgr_score, mean = {:.1f}, std = {:.1f}'.format(np.mean(results['initial_lgr_score']), np.std(results['initial_lgr_score'])))
    print('steps,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['steps']), np.std(results['steps'])))
    return np.mean(results['score'])

def callback(locals_, globals_, test_env):
    self_ = locals_['self']
    if not hasattr(self_, 'next_test_eval'):
        self_.next_test_eval = 0
    if self_.num_timesteps >= self_.next_test_eval:
        score = eval_pdefense_env(test_env, self_, 200, render_mode='none')
        summary = tf.Summary(value=[tf.Summary.Value(tag='score', simple_value=score)])
        locals_['writer'].add_summary(summary, self_.num_timesteps)
        self_.next_test_eval += 1000000
    return True

def train_helper(env_param, test_env_param, train_param, policy_fn, policy_param, directory, name):

    env = SubprocVecEnv([lambda: PDefenseEnv(
                            n_max_agents=env_param['n_max_agents'],
                            r_capture=env_param['r_capture'],
                            early_termination=env_param['early_termination'],
                            comm_adj_type=env_param['comm_adj_type'],
                            comm_adj_r=env_param.get('comm_adj_r', None)) for _ in range(train_param['n_env'])],
                        start_method='forkserver')

    test_env = PDefenseEnv(
        n_max_agents=test_env_param['n_max_agents'],
        r_capture=test_env_param['r_capture'],
        early_termination=test_env_param['early_termination'],
        comm_adj_type=env_param['comm_adj_type'],
        comm_adj_r=env_param.get('comm_adj_r', None))

    pkl_file = directory + '/' + name + '.pkl'
    tensorboard_log = directory + '/' + name

    if not os.path.exists(pkl_file):
        print('Creating new model ' + pkl_file + '.')
        model = PPO2(
            policy=policy_fn,
            policy_kwargs=policy_param,
            env=env,
            n_steps=train_param['n_steps'],
            ent_coef=0.001,
            verbose=1,
            tensorboard_log=tensorboard_log,
            full_tensorboard_log=False)
    else:
        print('Loading model ' + pkl_file + '.')
        model = PPO2.load(pkl_file, env,
            tensorboard_log=tensorboard_log)

    print('Learning...')
    model.learn(
        total_timesteps=train_param['total_timesteps'],
        log_interval=500,
        reset_num_timesteps=False,
        callback=functools.partial(callback, test_env=test_env))

    print('Saving model...')
    model.save(pkl_file)

    print('Finished.')

if __name__ == '__main__':

    import copy

    jobs = [] # string name, policy class, policy_kwargs

    # # Input feature processing with all parameters shared, msg_size = 0.
    # j = {}
    # j['policy'] = gnn_fwd.GnnFwd
    # j['policy_param'] = {
    #     'input_feat_layers':    (64,64),
    #     'feat_agg_layers':      (64,64),
    #     'msg_enc_layers':       (),
    #     'msg_size':             0,
    #     'msg_dec_layers':       (),
    #     'msg_agg_layers':       (),
    #     'pi_head_layers':       (),
    #     'vf_local_head_layers': (),
    #     'vf_global_head_layers':()}
    # j['name'] = j['policy'].policy_param_string(j['policy_param'])
    # jobs.append(j)

    # # Input feature processing with all parameters shared, msg_size = 0.
    # j = {}
    # j['policy'] = gnn_fwd.GnnFwd
    # j['policy_param'] = {
    #     'input_feat_layers':    (64,64),
    #     'feat_agg_layers':      (64,64),
    #     'msg_enc_layers':       (),
    #     'msg_size':             0,
    #     'msg_dec_layers':       (64,),
    #     'msg_agg_layers':       (64,),
    #     'pi_head_layers':       (),
    #     'vf_local_head_layers': (),
    #     'vf_global_head_layers':()}
    # j['name'] = j['policy'].policy_param_string(j['policy_param'])
    # jobs.append(j)

    # # Input feature processing with all parameters shared, msg_size = 8.
    # j = {}
    # j['policy'] = gnn_fwd.GnnFwd
    # j['policy_param'] = {
    #     'input_feat_layers':    (64,64),
    #     'feat_agg_layers':      (64,64),
    #     'msg_enc_layers':       (),
    #     'msg_size':             8,
    #     'msg_dec_layers':       (64,),
    #     'msg_agg_layers':       (64,),
    #     'pi_head_layers':       (),
    #     'vf_local_head_layers': (),
    #     'vf_global_head_layers':()}
    # j['name'] = j['policy'].policy_param_string(j['policy_param'])
    # jobs.append(j)

    # Miniature ICRA 2018 with msg_size = 8.
    # j = {}
    # j['policy'] = gnn_fwd.GnnFwd
    # j['policy_param'] = {
    #     'input_feat_layers':    (64,64),
    #     'feat_agg_layers':      (),
    #     'msg_enc_layers':       (64,64),
    #     'msg_size':             8,
    #     'msg_dec_layers':       (64,64),
    #     'msg_agg_layers':       (64,64),
    #     'pi_head_layers':       (),
    #     'vf_local_head_layers': (),
    #     'vf_global_head_layers':()}
    # j['name'] = j['policy'].policy_param_string(j['policy_param'])
    # jobs.append(j)

    # Miniature ICRA 2018 with msg_size = 0.
    # j = {}
    # j['policy'] = gnn_fwd.GnnFwd
    # j['policy_param'] = {
    #     'input_feat_layers':    (64,64),
    #     'feat_agg_layers':      (),
    #     'msg_enc_layers':       (64,64),
    #     'msg_size':             0,
    #     'msg_dec_layers':       (64,64),
    #     'msg_agg_layers':       (64,64),
    #     'pi_head_layers':       (),
    #     'vf_local_head_layers': (),
    #     'vf_global_head_layers':()}
    # j['name'] = j['policy'].policy_param_string(j['policy_param'])
    # jobs.append(j)

    # Miniature ICRA 2018 with msg_size = 8 and global vf head.
    j = {}
    j['policy'] = gnn_fwd.GnnFwd
    j['policy_param'] = {
        'input_feat_layers':    (64,64),
        'feat_agg_layers':      (),
        'msg_enc_layers':       (64,64),
        'msg_size':             8,
        'msg_dec_layers':       (64,64),
        'msg_agg_layers':       (64,64),
        'pi_head_layers':       (),
        'vf_local_head_layers': (),
        'vf_global_head_layers':(64,)}
    j['name'] = j['policy'].policy_param_string(j['policy_param'])
    jobs.append(j)

    # # Miniature ICRA 2018 with msg_size = 8 and global vf head.
    # j = {}
    # j['policy'] = gnn_fwd.GnnFwd
    # j['policy_param'] = {
    #     'input_feat_layers':    (128,128),
    #     'feat_agg_layers':      (),
    #     'msg_enc_layers':       (128,128),
    #     'msg_size':             8,
    #     'msg_dec_layers':       (128,128),
    #     'msg_agg_layers':       (128,128),
    #     'pi_head_layers':       (),
    #     'vf_local_head_layers': (),
    #     'vf_global_head_layers':(128,)}
    # j['name'] = j['policy'].policy_param_string(j['policy_param'])
    # jobs.append(j)


    env_param = {
        'n_max_agents':      9,
        'r_capture':         0.2,
        'early_termination': True,
        'comm_adj_type':     'circulant'
    }

    test_env_param = copy.deepcopy(env_param)
    test_env_param['early_termination'] = False

    train_param = {
        'n_env':32,
        'n_steps':32,
        'total_timesteps':50000000
    }

    root = 'models/2019-09-11/sigmoid/'

    for j in jobs:

        directory = root + '/' + env_param_string(env_param) + '_' + train_param_string(train_param)

        train_helper(
            env_param       = env_param,
            test_env_param  = test_env_param,
            train_param     = train_param,
            policy_fn       = j['policy'],
            policy_param    = j['policy_param'],
            directory       = directory,
            name            = j['name'])
