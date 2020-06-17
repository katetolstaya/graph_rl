import numpy as np
import gym
import gym_flock
import copy

import rl_comm.gnn_fwd as gnn_fwd
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.base_class import BaseRLModel
from rl_comm.ppo2 import PPO2

import rospy
from mav_manager.srv import Vec4Request, Vec4
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


if __name__ == '__main__':

    n_robots = 10
    x = np.zeros((n_robots, 2))
    names = ['quadrotor' + str(i + 1) for i in range(n_robots)]
    altitudes = np.linspace(start=3.0, stop=8.0, num=n_robots)

    rospy.init_node('gnn')
    r = rospy.Rate(10.0)

    def make_env():
        env_name = "CoverageFull-v0"
        my_env = gym.make(env_name)
        my_env = gym.wrappers.FlattenDictWrapper(my_env, dict_keys=my_env.env.keys)
        return my_env

    vec_env = SubprocVecEnv([make_env])

    # Specify pre-trained model checkpoint file.
    model_name = 'models/imitation_test/ckpt/ckpt_036.pkl'

    # load the dictionary of parameters from file
    model_params, params = BaseRLModel._load_from_file(model_name)

    new_model = PPO2(
        policy=gnn_fwd.GnnFwd,
        policy_kwargs=model_params['policy_kwargs'],
        env=vec_env)

    # update new model's parameters
    new_model.load_parameters(params)

    # N = 10
    model = new_model
    # render_mode = 'human'

    env = make_env()

    env.reset()
    # env.render(mode=render_mode)

    arl_env = env.env.env


    def state_callback(data, robot_index):
        x[robot_index, 0] = data.pose.position.x
        x[robot_index, 1] = data.pose.position.y


    for i, name in enumerate(names):
        topic_name = "/unity_ros/" + name + "/TrueState/pose"
        rospy.Subscriber(name=topic_name, data_class=PoseStamped, callback=state_callback, callback_args=i)

    services = [rospy.ServiceProxy("/" + name + "/mav_services/goTo", Vec4) for name in names]

    marker_publisher = rospy.Publisher('/planning_map/grid', MarkerArray, queue_size=100)


    def get_markers():
        marker_array = MarkerArray()

        for i in range(arl_env.n_agents):
            marker = Marker()
            marker.id = i
            marker.header.frame_id = "map"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.pose.orientation.w = 1.0

            if arl_env.robot_flag[i] == 1 and i < arl_env.n_robots:
                marker.pose.position.x = x[i, 0]
                marker.pose.position.y = x[i, 1]
                marker.pose.position.z = altitudes[i]
                rad = 6.0
                marker.color.a = 0.75
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.pose.position.x = arl_env.x[i, 0]
                marker.pose.position.y = arl_env.x[i, 1]
                marker.pose.position.z = 1.0
                marker.color.a = 1.0

                if arl_env.visited[i]:
                    rad = 2.0
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                else:
                    rad = 3.0
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0

            marker.scale.x = rad
            marker.scale.y = rad
            marker.scale.z = rad

            marker_array.markers.append(marker)

        return marker_array

    obs = env.reset()

    while True:

        marker_publisher.publish(get_markers())

        # update state and get new observation
        arl_env.update_state(x)
        obs, reward, _, _ = env.step(None)
        print(reward)

        action, states = model.predict(obs, deterministic=True)

        # env.render(mode=render_mode)

        next_loc = copy.copy(action.reshape((-1, 1)))

        # convert to next waypoint
        for i in range(arl_env.n_robots):
            next_loc[i] = arl_env.mov_edges[1][np.where(arl_env.mov_edges[0] == i)][action[i]]
        loc_commands = np.reshape(arl_env.x[next_loc, 0:2], (arl_env.n_robots, 2))

        # update last loc
        old_last_loc = arl_env.last_loc
        arl_env.last_loc = arl_env.closest_targets

        # send new waypoints
        for i, service in enumerate(services):
            goal_position = [loc_commands[i, 0], loc_commands[i, 1], altitudes[i], -1.57]
            goal_position = Vec4Request(goal_position)
            try:
                service(goal_position)
            except rospy.ServiceException:
                print("Service call failed")

        arl_env.last_loc = np.where(arl_env.last_loc == arl_env.closest_targets, old_last_loc, arl_env.last_loc)

        r.sleep()

