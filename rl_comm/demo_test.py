import numpy as np
from progress.bar import Bar

def eval_pdefense_env(env, model, N, render_mode='none'):
    """
    Evaluate a model against an environment over N games.
    """
    results = {
        'steps': np.zeros(N),
        'score': np.zeros(N),
        'lgr_score': np.zeros(N),
        'initial_lgr_score': np.zeros(N)}
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
            results['steps'][k] = info['steps']
            results['score'][k] = info['score']
            results['lgr_score'][k] = info['lgr_score']
            results['initial_lgr_score'][k] = info['initial_lgr_score']
            bar.next()
    return results

if __name__ == '__main__':

    from stable_baselines import PPO2
    from stable_baselines.common.vec_env import SubprocVecEnv

    from gym_pdefense.envs.pdefense_env_lgr import PDefenseEnv

    # Specify environment.
    env_param = {
        'n_max_agents':      9,
        'r_capture':         0.2,
        'early_termination': False,
        'comm_adj_type':     'circulant',
        'comm_adj_r':        None,
        'fov':               360
    }

    env = PDefenseEnv(
        n_max_agents=env_param['n_max_agents'],
        r_capture=env_param['r_capture'],
        early_termination=env_param['early_termination'],
        comm_adj_type=env_param['comm_adj_type'],
        comm_adj_r=env_param.get('comm_adj_r', None),
        fov=env_param['fov'])

    # Specify pre-trained model checkpoint file.
    model_name = 'ckpt_050.pkl'

    model = PPO2.load(model_name)

    print('\nPlay 100 games and return scores...')
    results = eval_pdefense_env(env, model, 1, render_mode='none')
    print('score,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['score']), np.std(results['score'])))
    print('init_lgr_score, mean = {:.1f}, std = {:.1f}'.format(np.mean(results['initial_lgr_score']), np.std(results['initial_lgr_score'])))
    print('steps,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['steps']), np.std(results['steps'])))
    print('')

    print('\nPlay games with live visualization...')
    eval_pdefense_env(env, model, 3, render_mode='human') # also support ffmpeg
