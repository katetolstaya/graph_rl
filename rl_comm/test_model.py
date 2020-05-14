import numpy as np
from progress.bar import Bar
import gym
import gym_flock
import time
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


def eval_model(env, model, N, render_mode='none'):
    """
    Evaluate a model against an environment over N games.
    """
    results = {'reward': np.zeros(N)}
    with Bar('Eval', max=N) as bar:
        for k in range(N):
            done = False
            obs = env.reset()
            # Run one game.
            while not done:
                action, states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = env.step(action)
                env.render(mode=render_mode)

                if render_mode == 'human':
                    time.sleep(0.1)

                # Record results.
                results['reward'][k] += rewards

            bar.next()
    return results


if __name__ == '__main__':
    env = make_env()
    vec_env = SubprocVecEnv([make_env])

    # Specify pre-trained model checkpoint file.
    # model_name = 'models/imitation_test/ckpt/ckpt_036.pkl'
    # model_name = 'models/imitation_20/ckpt/ckpt_013.pkl'
    # model_name = 'models/imitation_105/ckpt/ckpt_015.pkl'
    # model_name = 'models/imitation_77/ckpt/ckpt_127.pkl'
    # model_name = 'models/imitation_81/ckpt/ckpt_061.pkl'
    # model_name = 'models/rl_86/ckpt/ckpt_012.pkl'
    # model_name = 'models/rl_90/ckpt/ckpt_007.pkl'
    # model_name = 'models/rl_94/ckpt/ckpt_100.pkl'  # ent_coef  = 1e-5
    # model_name = 'models/rl_95/ckpt/ckpt_100.pkl'  # ent_coef  = 1e-6
    model_name = 'models/dagger_3/ckpt/ckpt_022.pkl'  # ent_coef  = 1e-6

    # load the dictionary of parameters from file
    model_params, params = BaseRLModel._load_from_file(model_name)

    policy_kwargs = model_params['policy_kwargs']

    # policy_kwargs = {
    #     # 'num_processing_steps': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],  #[1,1,1,1,1,1,3,3,3,1,1,1,1,1,1], #[1,1,1,3,1,1,3,1,1,1,3,1,1,1,1],
    #     'num_processing_steps': [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    #     'n_layers': 2,
    #     'latent_size': 16,
    # }

    new_model = PPO2(
        policy=gnn_fwd.GnnFwd,
        policy_kwargs=policy_kwargs,
        env=vec_env)

    # update new model's parameters
    new_model.load_parameters(params)

    print('\nPlay 10 games and return scores...')
    results = eval_model(env, new_model, 30, render_mode='none')
    print('reward,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['reward']), np.std(results['reward'])))
    print('')

    print('\nPlay games with live visualization...')
    eval_model(env, new_model, 10, render_mode='human')
