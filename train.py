import gym
import gym_pdefense

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

# multiprocess environment
n_cpu = 16
env = SubprocVecEnv([lambda: gym.make('PDefense-v0') for i in range(n_cpu)])

model = A2C(policy=MlpPolicy,
            env=env,
            n_steps=16,
            ent_coef=0.001,
            verbose=1)
model.learn(total_timesteps=10000000)
model.save("a2c_pdefense_1")
