import gym
import gym_pdefense

import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

env = gym.make('PDefense-v0')

model = A2C.load("a2c_pdefense_2_n_steps_16")

while True:
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(mode='human') # pick from ['human', 'ffmpeg']
    print('.')
