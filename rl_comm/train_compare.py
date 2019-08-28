import os.path
import multiprocessing

import gym
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C

from gym_pdefense.envs.pdefense_env import PDefenseEnv
import gnn_policies

def layers_string(layers):
    return '-'.join(str(l) for l in layers)

def policy_param_string(p):
    """Return identifier string for GnnCoord policy parameter dict."""
    return 'gnncoord_in_{inf}_ag_{ag}_pi_{pi}_vfl_{vfl}_vfg_{vfg}'.format(
        inf=layers_string(p['input_feat_layers']),
        ag= layers_string(p['feat_agg_layers']),
        pi= layers_string(p['pi_head_layers']),
        vfl=layers_string(p['vf_local_head_layers']),
        vfg=layers_string(p['vf_global_head_layers'])
        )

def train_param_string(p):
    """Return identifier string for A2C training parameter dict."""
    return 'ne_{ne}_ns_{ns}'.format(
        ne=p['n_env'],
        ns=p['n_steps'])

def env_param_string(p):
    """Return identifier string for PDefenseEnv environment parameter dict."""
    return 'na_{na}_rc_{rc}'.format(
        na=p['n_max_agents'],
        rc=p['r_capture'])

jobs = [] # string name, policy class, policy_kwargs

# MLP for fixed size games with separate pi/vf layers.
j = {}
j['policy'] = gnn_policies.MyMlpPolicy
j['policy_param'] = {}
j['name'] = 'mymlp'
jobs.append(j)

# Replicates MyMlpPolicy with separate pi/vf layers.
j = {}
j['policy'] = gnn_policies.OneNodePolicy
j['policy_param'] = {}
j['name'] = 'onenode'
jobs.append(j)

# In the special case of 1v1 games, replicates MyMlpPolicy with shared pi/vf layers.
j = {}
j['policy'] = gnn_policies.GnnCoord
j['policy_param'] = {
    'input_feat_layers':    (),
    'feat_agg_layers':      (64,64),
    'pi_head_layers':       (),
    'vf_local_head_layers': (),
    'vf_global_head_layers':()}
j['name'] = policy_param_string(j['policy_param'])
jobs.append(j)

# In the special case of 1v1 games, replicates MyMlpPolicy with separate pi/vf layers.
j = {}
j['policy'] = gnn_policies.GnnCoord
j['policy_param'] = {
    'input_feat_layers':    (),
    'feat_agg_layers':      (),
    'pi_head_layers':       (64,64),
    'vf_local_head_layers': (64,64),
    'vf_global_head_layers':()}
j['name'] = policy_param_string(j['policy_param'])
jobs.append(j)

# Input feature processing with all parameters shared.
j = {}
j['policy'] = gnn_policies.GnnCoord
j['policy_param'] = {
    'input_feat_layers':    (64,64),
    'feat_agg_layers':      (64,64),
    'pi_head_layers':       (),
    'vf_local_head_layers': (),
    'vf_global_head_layers':()}
j['name'] = policy_param_string(j['policy_param'])
jobs.append(j)

env_param = {
    'n_max_agents':3,
    'r_capture':   1.0
}

train_param = {
    'n_env':16,
    'n_steps':32
}

for j in jobs:

    # Single process version.
    # env = DummyVecEnv([lambda: PDefenseEnv(
    #     n_max_agents=env_param['n_max_agents'],
    #     r_capture=env_param['r_capture']) for _ in range(train_param['n_env'])])

    # Multi process version.
    env = SubprocVecEnv([lambda: PDefenseEnv(
        n_max_agents=env_param['n_max_agents'],
        r_capture=env_param['r_capture']) for _ in range(train_param['n_env'])])

    folder = 'test/' + env_param_string(env_param) + '_' + train_param_string(train_param)
    pkl_file = folder + '/' + j['name'] + '.pkl'
    tensorboard_log = './' + folder + '/' + j['name']

    if not os.path.exists(pkl_file):
        print('Creating new model ' + pkl_file + '.')
        model = A2C(
            policy=j['policy'],
            policy_kwargs=j['policy_param'],
            env=env,
            n_steps=train_param['n_env'],
            ent_coef=0.001,
            verbose=1,
            tensorboard_log=tensorboard_log,
            full_tensorboard_log=False)
    else:
        print('Loading model ' + pkl_file + '.')
        model = A2C.load(pkl_file, env,
            tensorboard_log=tensorboard_log)

    print('Learning...')
    model.learn(
        total_timesteps=1000000,
        log_interval = 500,
        reset_num_timesteps=False)
    print('Saving model...')
    model.save(pkl_file)
    print('Finished.')
