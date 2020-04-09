import numpy as np
import gym
import gym_flock
import time
import rl_comm.gnn_fwd as gnn_fwd
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.base_class import BaseRLModel

import copy
import rospy
from mav_manager.srv import Vec4Request, Vec4
from geometry_msgs.msg import PoseStamped
from stable_baselines import PPO2


def make_env():
    env_name = "MappingARLPartial-v0"
    # env_name = "MappingAirsim-v0"
    keys = ['nodes', 'edges', 'senders', 'receivers', 'step']
    env = gym.make(env_name)
    env = gym.wrappers.FlattenDictWrapper(env, dict_keys=keys)
    return env

if __name__ == '__main__':
    n_robots = 10
    x = np.zeros((n_robots, 2))
    names = ['quadrotor' + str(i + 1) for i in range(n_robots)]

    rospy.init_node('gnn')
    # TODO smaller rate here?
    r = rospy.Rate(10.0)

    altitudes = np.linspace(start=3.0, stop=8.0, num=n_robots)

    vec_env = SubprocVecEnv([make_env])

    # Specify pre-trained model checkpoint file.
    # model_name = 'models/diff7/diff7/ckpt/ckpt_034.pkl'
    # model_name = 'models/cross5/cross5/ckpt/ckpt_001.pkl'
    # model_name = 'models/2020-01-20/2020-01-20/ckpt/ckpt_002.pkl'
    # model_name = 'models/disc/disc/ckpt/ckpt_000.pkl'
    # model_name = 'ckpt_000.pkl'
    # model_name = 'models/newnew/newnew/ckpt/ckpt_000.pkl'
    # model_name = 'models/rec/rec/ckpt/ckpt_067.pkl'
    # model_name = 'models/feat32/feat32/ckpt/ckpt_020.pkl'
    # policy_param = {'num_processing_steps': 5}
    model_name = 'models/feat3275/feat3275/ckpt/ckpt_041.pkl'
    policy_param = {}
    n_steps = 32

    new_model = PPO2(
        policy=gnn_fwd.GnnFwd,
        policy_kwargs=policy_param,
        env=vec_env,
        learning_rate=1e-6,
        cliprange=1.0,
        n_steps=n_steps,
        ent_coef=0.0001,
        vf_coef=0.5,
        verbose=1,
        full_tensorboard_log=False)

    # load the dictionary of parameters from file
    _, params = BaseRLModel._load_from_file(model_name)

    # update new model's parameters
    new_model.load_parameters(params)

    N = 10
    model = new_model
    render_mode = 'human'

    env = make_env()

    env.reset()

    arl_env = env.env.env


    def state_callback(data, robot_index):
        x[robot_index, 0] = data.pose.position.x
        x[robot_index, 1] = data.pose.position.y


    for i, name in enumerate(names):
        topic_name = "/unity_ros/" + name + "/TrueState/pose"
        rospy.Subscriber(name=topic_name, data_class=PoseStamped, callback=state_callback, callback_args=i)

    services = [rospy.ServiceProxy("/" + name + "/mav_services/goTo", Vec4) for name in names]

    rewards = [0] * N

    for k in range(N):
        done = False
        obs = env.reset()
        # Run one game.
        while not done:
            # update state and get new observation
            arl_env.update_state(x)
            obs, reward, _, _ = env.step(None)
            print(reward)
            rewards[k] += reward

            action, states = model.predict(obs, deterministic=True)

            env.render(mode=render_mode)

            next_loc = copy.copy(action.reshape((-1, 1)))

            # convert to next waypoint
            for i in range(arl_env.n_robots):
                next_loc[i] = arl_env.mov_edges[1][np.where(arl_env.mov_edges[0] == i)][action[i]]
            loc_commands = np.reshape(arl_env.x[next_loc, 0:2], (arl_env.n_robots, 2))

            # update last loc
            # TODO does this go here or before update state/recompute graph?
            old_last_loc = arl_env.last_loc
            arl_env.last_loc = arl_env.closest_targets

            # send new waypoints
            for i, service in enumerate(services):
                # TODO convert GNN output to next location
                goal_position = [loc_commands[i, 0], loc_commands[i, 1], altitudes[i], -1.57]
                # goal_position = [x[i, 0]+0.1, x[i, 1], 5.0, -1.57]
                goal_position = Vec4Request(goal_position)
                try:
                    service(goal_position)
                except rospy.ServiceException:
                    print("Service call failed")

            arl_env.last_loc = np.where(arl_env.last_loc == arl_env.closest_targets, old_last_loc, arl_env.last_loc)

            r.sleep()
        print(rewards[k])

    print('reward,          mean = {:.1f}, std = {:.1f}'.format(np.mean(rewards), np.std(rewards)))
    print('')
