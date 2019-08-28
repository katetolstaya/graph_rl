from stable_baselines import A2C

from gym_pdefense.envs.pdefense_env import PDefenseEnv
import gnn_policies

env_param = {
    'n_max_agents':3,
    'r_capture':   1.0
}

model_name = 'test/na_3_rc_1.0_ne_16_ns_32/gnncoord_in_64-64_ag_64-64_pi__vfl__vfg_.pkl'

env = PDefenseEnv(
    n_max_agents=env_param['n_max_agents'],
    r_capture=env_param['r_capture'])

model = A2C.load(model_name)

while True:
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(mode='human') # pick from ['human', 'ffmpeg']
    print(env.step_count)
