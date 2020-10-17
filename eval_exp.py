import numpy as np
import gym
import gym_flock
import glob
import sys
import rl_comm.gnn_fwd as gnn_fwd
from rl_comm.ppo2 import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.base_class import BaseRLModel


def make_env():
    # env_name = "CoverageFull-v0"
    env_name = "CoverageARL-v0"
    my_env = gym.make(env_name)
    my_env = gym.wrappers.FlattenDictWrapper(my_env, dict_keys=my_env.env.keys)
    return my_env


def eval_model(env, model, n_episodes):
    """
    Evaluate a model against an environment over N games.
    """
    results = {'reward': np.zeros(n_episodes)}
    for k in range(n_episodes):
        done = False
        obs = env.reset()
        # Run one game.
        while not done:
            action, states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            results['reward'][k] += rewards
    return results


def load_model(model_name, vec_env, new_model=None):
    model_params, params = BaseRLModel._load_from_file(model_name)

    if new_model is None:
        new_model = PPO2(
            policy=gnn_fwd.MultiGnnFwd,
            policy_kwargs=model_params['policy_kwargs'],
            env=vec_env)

    # update new model's parameters
    new_model.load_parameters(params)
    return new_model


if __name__ == '__main__':
    fname = sys.argv[1]

    env = make_env()
    vec_env = SubprocVecEnv([make_env])

    ckpt_dir = 'models/' + fname + '/ckpt'
    new_model = None

    try:
        ckpt_list = sorted(glob.glob(str(ckpt_dir) + '/*.pkl'))
        ckpt_idx = int(ckpt_list[-2][-7:-4])
    except IndexError:
        print('Invalid experiment folder name!')
        raise

    best_score = -np.Inf
    best_idx = 0

    for i in range(0, ckpt_idx, 5):
        model_name = ckpt_dir + '/ckpt_' + str(i).zfill(3) + '.pkl'
        new_model = load_model(model_name, vec_env, new_model)
        results = eval_model(env, new_model, 25)
        new_score = np.mean(results['reward'])
        print('Testing ' + model_name + ' : ' + str(new_score))

        if new_score > best_score:
            best_score = new_score
            best_idx = i

    model_name = ckpt_dir + '/ckpt_' + str(best_idx).zfill(3) + '.pkl'
    new_model = load_model(model_name, vec_env, new_model)
    n_episodes = 100
    results = eval_model(env, new_model, n_episodes)
    mean_reward = np.mean(results['reward'])
    std_reward = np.std(results['reward'])
    print('Reward over {} episodes: mean = {:.1f}, std = {:.1f}'.format(n_episodes, mean_reward, std_reward))
    quit()
