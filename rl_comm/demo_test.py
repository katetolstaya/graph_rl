import numpy as np
from progress.bar import Bar
import gym
import gym_flock
import time
# from rl_comm.train import policy_fn
import rl_comm.gnn_fwd as gnn_fwd
import tensorflow as tf


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

                if render_mode == 'human':
                    time.sleep(0.5)

                # Record results.
                results['reward'][k] += rewards

            bar.next()
    return results


if __name__ == '__main__':
    from stable_baselines import PPO2

    env = make_env()

    # Specify pre-trained model checkpoint file.
    model_name = 'models/2019-09-13/2019-09-22/ckpt/ckpt_050.pkl'

    model = PPO2.load(model_name)

    policy_param = {'num_processing_steps': 7}
    n_steps = 32
    new_model = PPO2(
        policy=gnn_fwd.GnnFwd,
        policy_kwargs=policy_param,
        env=env,
        learning_rate=1e-6,
        cliprange=1.0,
        n_steps=n_steps,
        ent_coef=0.0001,
        vf_coef=0.5,
        verbose=1,
        full_tensorboard_log=False)

    # copy the policy weights
    copy_weights = tf.group(*[va.assign(vb) for va, vb in
                              zip(new_model.policy.policy_model.variables, model.policy.policy_model.variables)])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(copy_weights)


    print('\nPlay 100 games and return scores...')
    results = eval_model(env, new_model, 1, render_mode='none')
    print('reward,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['reward']), np.std(results['reward'])))
    print('')

    print('\nPlay games with live visualization...')
    eval_model(env, new_model, 3, render_mode='human')  # also support ffmpeg
