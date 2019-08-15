import os.path

import gym
import gym_pdefense

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

# multiprocess environment
n_cpu = 16
env = SubprocVecEnv([lambda: gym.make('PDefense-v0') for i in range(n_cpu)])

name = 'a2c_pdefense_2_n_steps_16'
pkl_file = name + '.pkl'
tensorboard_log = './' + name + '_tensorboard/'

if not os.path.exists(pkl_file):
    print('Creating new model ' + pkl_file + '.')
    model = A2C(
        policy=MlpPolicy,
        env=env,
        n_steps=16,
        ent_coef=0.001,
        verbose=1,
        tensorboard_log=tensorboard_log)
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
