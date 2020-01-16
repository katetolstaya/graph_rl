import warnings
from typing import Dict
import numpy as np
import gym
import gym_flock
from gym import spaces

from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.base_class import _UnvecWrapper
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize


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

    is_vec_env = False
    if isinstance(env, VecEnv) and not isinstance(env, _UnvecWrapper):
        is_vec_env = True
        if env.num_envs > 1:
            warnings.warn("You are using multiple envs, only the data from the first one will be recorded.")

    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

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
        observations.append(obs)

        action = env.env.env.controller()

        obs, reward, done, _ = env.step(action)

        # Use only first env
        if is_vec_env:
            action = np.array([action[0]])
            reward = np.array([reward[0]])
            done = np.array([done[0]])

        actions.append(action)
        rewards.append(reward)
        episode_starts.append(done)
        reward_sum += reward
        idx += 1
        if done:
            if not is_vec_env:
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


env_name = "MappingRad1-v0"


def make_env():
    keys = ['nodes', 'edges', 'senders', 'receivers']
    env = gym.make(env_name)
    env = gym.wrappers.FlattenDictWrapper(env, dict_keys=keys)
    return env


# env = VecNormalize(SubprocVecEnv([make_env] * 1), norm_obs=False, norm_reward=True)
generate_expert_traj(env=make_env(), save_path='expert_rad2', n_episodes=1000)