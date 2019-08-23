import os.path

import gym
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

import gym_pdefense
import gnn_policies

policy = gnn_policies.GnnCoord

# multiprocess environment
n_cpu = 16
env = SubprocVecEnv([lambda: gym.make('PDefense-v0') for i in range(n_cpu)])

name = 'a2c_pdefense_2_gnncoord_n_steps_16'
pkl_file = name + '.pkl'
tensorboard_log = './' + name + '_tensorboard/'

if not os.path.exists(pkl_file):
    print('Creating new model ' + pkl_file + '.')
    model = A2C(
        policy=policy,
        env=env,
        n_steps=32,
        ent_coef=0.001,
        verbose=1,
        tensorboard_log=tensorboard_log)
else:
    print('Loading model ' + pkl_file + '.')
    model = A2C.load(pkl_file, env,
        tensorboard_log=tensorboard_log)

print('Learning...')
model.learn(
    total_timesteps=40000000,
    log_interval = 500,
    reset_num_timesteps=False)
print('Saving model...')
model.save(pkl_file)
print('Finished.')
