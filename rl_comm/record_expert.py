from typing import Dict
import numpy as np
import gym
import gym_flock
from gym import spaces
import sys

def generate_expert_traj(env, save_path=None, n_episodes=1000):
    """
    Train expert controller (if needed) and record expert trajectories.

    .. note::

        only Box and Discrete spaces are supported for now.

    :param env: (gym.Env) The environment, if not defined then it tries to use the model
        environment.
    :param save_path: (str) Path without the extension where the expert dataset will be saved
        (ex: 'expert_cartpole' -> creates 'expert_cartpole.npz').
        If not specified, it will not save, and just return the generated expert trajectories.
        This parameter must be specified for image-based environments.
    :param n_episodes: (int) Number of trajectories (episodes) to record
    :return: (dict) the generated expert trajectories.
    """

    assert env is not None, "You must set the env in the model or pass it to the function."

    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete) or
            isinstance(env.action_space, spaces.MultiDiscrete)), "Action space type not supported"

    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    idx = 0

    while ep_idx < n_episodes:

        try:
            action = env.env.env.controller(random=False, greedy=False)
        except AssertionError:
            obs = env.reset()
            reward_sum = 0.0
            continue

        observations.append(obs)
        actions.append(action)

        obs, reward, done, _ = env.step(action)

        rewards.append(reward)
        episode_starts.append(done)
        reward_sum += reward
        idx += 1

        if done:
            print(ep_idx)
            obs = env.reset()

            episode_returns[ep_idx] = reward_sum
            reward_sum = 0.0
            ep_idx += 1

    if isinstance(env.observation_space, spaces.Box):
        observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    elif isinstance(env.observation_space, spaces.Discrete):
        observations = np.array(observations).reshape((-1, 1))

    if isinstance(env.action_space, spaces.Box):
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, spaces.Discrete):
        actions = np.array(actions).reshape((-1, 1))
    elif isinstance(env.action_space, spaces.MultiDiscrete):
        actions = np.array(actions).reshape((-1, len(env.action_space.nvec)))

    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])

    assert len(observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }  # type: Dict[str, np.ndarray]

    for key, val in numpy_dict.items():
        print(key, val.shape)

    if save_path is not None:
        np.savez(save_path, **numpy_dict)

    env.close()

    return numpy_dict


env_name = "MappingRad-v0"


def make_env():
    keys = ['nodes', 'edges', 'senders', 'receivers', 'step']
    env = gym.make(env_name)
    env = gym.wrappers.FlattenDictWrapper(env, dict_keys=keys)
    return env



if len(sys.argv) >= 2:
    name = sys.argv[1]
else:
    name = 'feat'

# generate_expert_traj(env=make_env(), save_path='data/disc7', n_episodes=1000)
generate_expert_traj(env=make_env(), save_path='data/' + name, n_episodes=5000)
