import numpy as np


def print_key_if_true(dictionary, key):
    """
    Print each key string whose value in dictionary is True.
    """
    if dictionary[key]:
        return key + ', '
    return ''


def eval_pdefense_env(env, model, N, render_mode='none'):
    """
    Evaluate a model against an environment over N games.
    """
    results = {
        'steps': np.zeros(N),
        'score': np.zeros(N),
        'lgr_score': np.zeros(N),
        'initial_lgr_score': np.zeros(N)
    }

    for k in range(N):
        done = False
        obs = env.reset()
        # Run one game.
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            env.render(mode=render_mode)  # pick from ['none', human', 'ffmpeg']

        # Display results.
        cause = ''.join([print_key_if_true(info, key) for key in
                         ['all_agents_dead', 'all_targets_dead', 'lgr_score_increased', 'no_more_rewards']])
        print('{:>2} {}={}+{} {}'.format(
            info['steps'],
            info['initial_lgr_score'],
            info['score'],
            info['lgr_score'],
            cause))

        # Record results.
        results['steps'][k] = info['steps']
        results['score'][k] = info['score']
        results['lgr_score'][k] = info['lgr_score']
        results['initial_lgr_score'][k] = info['initial_lgr_score']

    print()
    print('score,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['score']), np.std(results['score'])))
    print('init_lgr_score, mean = {:.1f}, std = {:.1f}'.format(np.mean(results['initial_lgr_score']),
                                                               np.std(results['initial_lgr_score'])))
    return np.mean(results['score'])


if __name__ == '__main__':
    from stable_baselines import A2C
    from stable_baselines.common.vec_env import SubprocVecEnv

    from gym_pdefense.envs.pdefense_env import PDefenseEnv

    # Specify environment.
    env_param = {
        'n_max_agents': 9,
        'r_capture': 0.2,
        'early_termination': False
    }

    env = PDefenseEnv(
        n_max_agents=env_param['n_max_agents'],
        r_capture=env_param['r_capture'],
        early_termination=env_param['early_termination'])

    # Specify model.
    # model_name = 'models/2019-09-07/ppo2_lgr_9v9/na_9_rc_0.2_ne_32_ns_32/gnnfwd_in_64-64_ag__enc_64-64_msg_8_dec_64-64_ag_64-64_pi__vfl__vfg_64.pkl'
    model_name = 'models/2019-09-10/debug/na_9_rc_0.2_ne_32_ns_32/gnnfwd_in_64-64_ag__enc_64-64_msg_8_dec_64-64_ag_64-64_pi__vfl__vfg_64.pkl'

    model = A2C.load(model_name)

    # Evaluate.
    eval_pdefense_env(env, model, 100, render_mode='none')
