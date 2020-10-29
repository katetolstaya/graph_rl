import numpy as np
import gym
import gym_flock
import glob
import sys
import rl_comm.gnn_fwd as gnn_fwd
from rl_comm.ppo2 import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.base_class import BaseRLModel
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def make_env():
    env_name = "CoverageARL-v1"
    my_env = gym.make(env_name)
    my_env = gym.wrappers.FlattenDictWrapper(my_env, dict_keys=my_env.env.keys)
    return my_env


diameters = [15,20,25,30,40,50,60,70,80,90,100]

def eval_model(env, model, n_episodes):
    """
    Evaluate a model against an environment over N games.
    """
    results = {'reward': np.zeros(n_episodes), 'diameter': np.zeros(n_episodes)}
    for k in range(n_episodes):

        env.env.env.subgraph_size = env.env.env.range_xy / np.random.uniform(1.9, 4.0)
        done = False
        obs = env.reset()

        env.env.env.controller(random=False, greedy=True)
        diameter = env.env.env.graph_diameter
        print(diameter)
        if diameter in diameters:
            results['diameter'][k] = env.env.env.graph_diameter
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
    fnames = ['nl2_1_3_16', 'nl2_1_19_16']
    labels = ['K = 3', 'K = 19']
    colors = ['tab:blue', 'tab:orange']

    env = make_env()
    vec_env = SubprocVecEnv([make_env])
    fig = plt.figure()
    fig = plt.figure(figsize=(6, 4))

    for fname, label, color in zip(fnames, labels, colors):
        print('Evaluating ' + fname)

        ckpt_dir = 'models/' + fname + '/ckpt'
        new_model = None

        try:
            ckpt_list = sorted(glob.glob(str(ckpt_dir) + '/*.pkl'))
            ckpt_idx = int(ckpt_list[-2][-7:-4])
        except IndexError:
            print('Invalid experiment folder name!')
            raise

        model_name = ckpt_dir + '/ckpt_' + str(ckpt_idx).zfill(3) + '.pkl'
        new_model = load_model(model_name, vec_env, None)
        n_episodes = 2000
        results = eval_model(env, new_model, n_episodes)

        # x = results['diameter']
        # y = results['reward']
        # z = np.polyfit(results['diameter'], results['reward'], 1)
        # f = np.poly1d(z)
        # x_new = np.linspace(np.min(x), np.max(x), 50)
        # y_new = f(x_new)
        # plt.plot(x, y, 'o', x_new, y_new, label=label, color=color)

        means = []
        sems = []
        cur_diameters = []
        for d in diameters:
            rewards = results['reward'][results['diameter'] == d][0:10]
            if len(rewards) > 0:
                means.append(np.mean(rewards))
                sems.append(np.std(rewards)/np.sqrt(len(rewards)))
                cur_diameters.append(d)

        plt.errorbar(cur_diameters, means, yerr=sems, label=label)

        mean_reward = np.mean(results['reward'])
        std_reward = np.std(results['reward'])
        print('Reward over {} episodes: mean = {:.1f}, std = {:.1f}'.format(n_episodes, mean_reward, std_reward))
    plt.xlabel('Graph Diameter')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.savefig('field2.eps', format='eps')
    plt.show()
