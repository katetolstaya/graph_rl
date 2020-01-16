import numpy as np
from progress.bar import Bar
import gym
import gym_flock


def make_env():
    keys = ['nodes', 'edges', 'senders', 'receivers']
    env = gym.make("MappingRad1-v0")
    env = gym.wrappers.FlattenDictWrapper(env, dict_keys=keys)
    return env


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

                # Record results.
                results['reward'][k] += rewards

            bar.next()
    return results


if __name__ == '__main__':
    from stable_baselines import PPO2
    env = make_env()

    # Specify pre-trained model checkpoint file.
    model_name = 'models/2019-09-13/2019-09-22/ckpt/ckpt_002.pkl'

    model = PPO2.load(model_name)

    print('\nPlay 100 games and return scores...')
    results = eval_model(env, model, 1, render_mode='none')
    print('reward,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['reward']), np.std(results['reward'])))
    print('')

    print('\nPlay games with live visualization...')
    eval_model(env, model, 3, render_mode='human')  # also support ffmpeg
