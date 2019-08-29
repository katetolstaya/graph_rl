import os.path

from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C

from gym_pdefense.envs.pdefense_env import PDefenseEnv
import gnn_obs
import gnn_fwd

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

# Input feature processing with all parameters shared, no messages.
j = {}
j['policy'] = gnn_obs.GnnObs
j['policy_param'] = {
    'input_feat_layers':    (64,64),
    'feat_agg_layers':      (64,64),
    'pi_head_layers':       (),
    'vf_local_head_layers': (),
    'vf_global_head_layers':()}
j['name'] = j['policy'].policy_param_string(j['policy_param'])
jobs.append(j)

# Input feature processing with all parameters shared, msg_size = 0.
j = {}
j['policy'] = gnn_fwd.GnnFwd
j['policy_param'] = {
    'input_feat_layers':    (64,64),
    'feat_agg_layers':      (64,64),
    'msg_enc_layers':       (),
    'msg_size':             0,
    'msg_dec_layers':       (),
    'msg_agg_layers':       (),
    'pi_head_layers':       (),
    'vf_local_head_layers': (),
    'vf_global_head_layers':()}
j['name'] = j['policy'].policy_param_string(j['policy_param'])
jobs.append(j)

env_param = {
    'n_max_agents':3,
    'r_capture':   0.5
}

train_param = {
    'n_env':16,
    'n_steps':32
}

root = 'debug'

for j in jobs:

    # Multi process version.
    env = SubprocVecEnv([lambda: PDefenseEnv(
        n_max_agents=env_param['n_max_agents'],
        r_capture=env_param['r_capture']) for _ in range(train_param['n_env'])])

    folder = root + '/' + env_param_string(env_param) + '_' + train_param_string(train_param)
    pkl_file = folder + '/' + j['name'] + '.pkl'
    tensorboard_log = folder + '/' + j['name']

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
        total_timesteps=10000000,
        log_interval = 500,
        reset_num_timesteps=False)
    print('Saving model...')
    model.save(pkl_file)
    print('Finished.')
