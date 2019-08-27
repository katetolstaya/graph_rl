import os.path
import multiprocessing

import gym
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

from gym_pdefense.envs.pdefense_env import PDefenseEnv
import gnn_policies

# name = 'a2c_pdefense_2_mymlp_n_steps_16'
# policy = gnn_policies.MyMlpPolicy

# name = 'a2c_pdefense_2_onenode_n_steps_16'
# policy = gnn_policies.OneNodePolicy

# name = 'a2c_pdefense_2_gnncoord_n_steps_16'
# policy = gnn_policies.GnnCoord

jobs = []
# jobs.append(('a2c_pdefense_2_mymlp_n_steps_16',     gnn_policies.MyMlpPolicy))
# jobs.append(('a2c_pdefense_2_onenode_n_steps_16',   gnn_policies.OneNodePolicy))
jobs.append(('a2c_pdefense_2_gnncoord_n_steps_16_2x64_new',  gnn_policies.GnnCoord))

for name, policy in jobs:

    n_env = 16
    env = SubprocVecEnv([lambda: PDefenseEnv(n_max_agents=2) for i in range(n_env)])

    folder = 'compare'
    pkl_file = folder + '/' + name + '.pkl'
    tensorboard_log = './' + folder + '/' + name + '_tb/'

    if not os.path.exists(pkl_file):
        print('Creating new model ' + pkl_file + '.')
        model = A2C(
            policy=policy,
            env=env,
            n_steps=16,
            ent_coef=0.001,
            verbose=1,
            tensorboard_log=tensorboard_log,
            full_tensorboard_log=True)
    else:
        print('Loading model ' + pkl_file + '.')
        model = A2C.load(pkl_file, env,
            tensorboard_log=tensorboard_log)

    print('Learning...')
    model.learn(
        total_timesteps=100000,
        log_interval = 500,
        reset_num_timesteps=False)
    print('Saving model...')
    model.save(pkl_file)
    print('Finished.')
