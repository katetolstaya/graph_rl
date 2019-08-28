import os.path
import multiprocessing

import gym
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

from gym_pdefense.envs.pdefense_env import PDefenseEnv
import gnn_policies

jobs = [] # string name, policy class, policy_kwargs

jobs.append({
    'name':'mymlp',
    'policy':gnn_policies.MyMlpPolicy,
    'policy_kwargs':{}
    })

jobs.append({
    'name':'onenode',
    'policy':gnn_policies.OneNodePolicy,
    'policy_kwargs':{}
    })

# jobs.append({
#     'name':'gnncoord_in_64-64_ag_64-64_pi__vf__',
#     'policy':gnn_policies.GnnCoord,
#     'policy_kwargs':{
#         'input_feat_layers':(64,64),
#         'feat_agg_layers':(64,64),
#         'pi_head_layers':(),
#         'vf_head_layers':()}
#     })

jobs.append({
    'name':'gnncoord_in__ag_64-64_pi__vf__',
    'policy':gnn_policies.GnnCoord,
    'policy_kwargs':{
        'input_feat_layers':(),
        'feat_agg_layers':(64,64),
        'pi_head_layers':(),
        'vf_head_layers':()}
    })

jobs.append({
    'name':'gnncoord_in__ag__pi_64-64_vf_64-64',
    'policy':gnn_policies.GnnCoord,
    'policy_kwargs':{
        'input_feat_layers':(),
        'feat_agg_layers':(),
        'pi_head_layers':(64,64),
        'vf_head_layers':(64,64)}
    })


for j in jobs:

    n_env = 16
    env = SubprocVecEnv([lambda: PDefenseEnv(n_max_agents=2) for i in range(n_env)])

    folder = 'agents_2_steps_32_replicate'
    pkl_file = folder + '/' + j['name'] + '.pkl'
    tensorboard_log = './' + folder + '/' + j['name'] + '_tb/'

    if not os.path.exists(pkl_file):
        print('Creating new model ' + pkl_file + '.')
        model = A2C(
            policy=j['policy'],
            policy_kwargs=j['policy_kwargs'],
            env=env,
            n_steps=32,
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
        total_timesteps=30000000,
        log_interval = 500,
        reset_num_timesteps=False)
    print('Saving model...')
    model.save(pkl_file)
    print('Finished.')
