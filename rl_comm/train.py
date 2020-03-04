import gym
import gym_flock
import functools
import glob
import sys
from pathlib import Path
from stable_baselines.common import BaseRLModel
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.gail import ExpertDataset

import rl_comm.gnn_fwd as gnn_fwd
from rl_comm.ppo2 import PPO2
from rl_comm.utils import ckpt_file, callback


def train_helper(env_param, test_env_param, train_param, policy_fn, policy_param, directory):
    save_dir = Path(directory)
    tb_dir = save_dir / 'tb'
    ckpt_dir = save_dir / 'ckpt'
    for d in [save_dir, tb_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    env = SubprocVecEnv([env_param['make_env']] * train_param['n_env'])
    test_env = SubprocVecEnv([test_env_param['make_env']])

    if train_param['use_checkpoint']:
        # Find latest checkpoint index.
        ckpt_list = sorted(glob.glob(str(ckpt_dir) + '/*.pkl'))
        if len(ckpt_list) == 0:
            ckpt_idx = None
        else:
            ckpt_idx = int(ckpt_list[-1][-7:-4])
    else:
        ckpt_idx = None

    # Load or create model.
    if ckpt_idx is not None:
        print('\nLoading model {}.\n'.format(ckpt_file(ckpt_dir, ckpt_idx).name))
        model = PPO2.load(str(ckpt_file(ckpt_dir, ckpt_idx)), env, tensorboard_log=str(tb_dir))
        ckpt_idx += 1
    else:
        print('\nCreating new model.\n')
        model = PPO2(
            policy=policy_fn,
            policy_kwargs=policy_param,
            env=env,
            learning_rate=train_param['train_lr'],
            cliprange=1.0,
            n_steps=train_param['n_steps'],
            ent_coef=0.01,
            vf_coef=0.5,
            verbose=1,
            tensorboard_log=str(tb_dir),
            full_tensorboard_log=False)
        ckpt_idx = 0

        if 'load_trained_policy' in train_param and train_param['load_trained_policy'] is not None:
            model_name = train_param['load_trained_policy']

            # load the dictionary of parameters from file
            _, params = BaseRLModel._load_from_file(model_name)

            # update new model's parameters
            model.load_parameters(params)

    if 'pretrain_dataset' in train_param and train_param['pretrain_dataset'] is not None:

        dataset = ExpertDataset(expert_path=train_param['pretrain_dataset'], traj_limitation=-1, batch_size=train_param['pretrain_batch'], randomize=True)
        model.pretrain(dataset, n_epochs=train_param['pretrain_epochs'], learning_rate=train_param['pretrain_lr'],
                       val_interval=1, test_env=test_env, adam_epsilon=train_param['pretrain_adam_eps'])

        model.save(str(ckpt_file(ckpt_dir, ckpt_idx)))
        del dataset
        ckpt_idx += 1

    # Training loop.
    print('\nBegin training.\n')
    while model.num_timesteps <= train_param['total_timesteps']:
        print('\nLearning...\n')
        model.learn(
            total_timesteps=train_param['checkpoint_timesteps'],
            log_interval=500,
            reset_num_timesteps=False,
            callback=functools.partial(callback, test_env=test_env))

        print('\nSaving model {}.\n'.format(ckpt_file(ckpt_dir, ckpt_idx).name))
        model.save(str(ckpt_file(ckpt_dir, ckpt_idx)))
        ckpt_idx += 1

    print('Finished.')


def main():
    jobs = []  # string name, policy class, policy_kwargs

    j = {}
    j['policy'] = gnn_fwd.GnnFwd
    j['policy_param'] = {'num_processing_steps': 5}  #[1, 1, 2, 2, 2]}
    # j['name'] = j['policy'].policy_param_string(j['policy_param'])

    if len(sys.argv) >= 2:
        j['name'] = sys.argv[1]
    else:
        j['name'] = 'rad'

    jobs.append(j)

    env_name = "MappingRad-v0"

    def make_env():
        keys = ['nodes', 'edges', 'senders', 'receivers', 'step']
        env = gym.make(env_name)
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=keys)
        return env

    env_param = {'make_env': make_env}
    test_env_param = {'make_env': make_env}

    train_param = {
        'n_env': 16,
        'n_steps': 10,
        'checkpoint_timesteps': 100000,
        # 'total_timesteps': 50000000,
        'total_timesteps': 1,
        # 'total_timesteps': 0,
        # 'load_trained_policy': None,  # 'ckpt_026.pkl'
        # 'load_trained_policy': "models/enc/enc/ckpt/ckpt_000.pkl",
        'pretrain_dataset': 'data/disc6.npz',
        # 'pretrain_dataset': None,
        'pretrain_epochs': 100,
        'pretrain_batch': 20,
        'pretrain_lr': 1e-6,
        # 'pretrain_lr': 1e-5,
        # 'pretrain_lr': 1e-7,
        # 'pretrain_adam_eps': 1e-4,
        # 'pretrain_adam_eps': 1e-6,
        'pretrain_adam_eps': 1e-8,
        # 'train_lr': 1e-7,
        'train_lr': 1e-8,
        'use_checkpoint': False,
    }
    # 'pretrain_dataset' = 'data/expert_rad2.npz'
    # 'pretrain_epochs' = 5000
    # 'pretrain_lr' = 1e-6

    root = Path('models/' + j['name'])

    for j in jobs:
        directory = root / j['name']  # env_param_string(env_param) / train_param_string(train_param) / j['name']

        train_helper(
            env_param=env_param,
            test_env_param=test_env_param,
            train_param=train_param,
            policy_fn=j['policy'],
            policy_param=j['policy_param'],
            directory=directory)


if __name__ == '__main__':
    main()
