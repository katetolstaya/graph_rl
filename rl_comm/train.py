import gym
import gym_flock
import configparser
import json
from os import path
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


def train_helper(env_param, test_env_param, train_param, pretrain_param, policy_fn, policy_param, directory):
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
            cliprange=train_param['cliprange'],
            adam_epsilon=train_param['adam_epsilon'],
            n_steps=train_param['n_steps'],
            verbose=1,
            tensorboard_log=str(tb_dir),
            full_tensorboard_log=False)
        ckpt_idx = 0

        if 'load_trained_policy' in train_param and len(train_param['load_trained_policy']) > 0:
            model_name = train_param['load_trained_policy']

            # load the dictionary of parameters from file
            _, params = BaseRLModel._load_from_file(model_name)

            # update new model's parameters
            model.load_parameters(params)

    if pretrain_param is not None:
        ckpt_params = {
            'ckpt_idx': ckpt_idx,
            'ckpt_epochs': pretrain_param['pretrain_checkpoint_epochs'],
            'ckpt_file': ckpt_file,
            'ckpt_dir': ckpt_dir
        }

        dataset = ExpertDataset(expert_path=pretrain_param['pretrain_dataset'], traj_limitation=-1,
                                batch_size=pretrain_param['pretrain_batch'], randomize=True)

        model.pretrain(dataset, n_epochs=pretrain_param['pretrain_epochs'],
                       learning_rate=pretrain_param['pretrain_lr'],
                       val_interval=1, test_env=test_env, ckpt_params=ckpt_params)

        del dataset
        ckpt_idx += int(train_param['pretrain_epochs'] / ckpt_params['pretrain_checkpoint_epochs'])

    # Training loop.
    print('\nBegin training.\n')
    while train_param['total_timesteps'] > 0 and model.num_timesteps <= train_param['total_timesteps']:
        print('\nLearning...\n')
        model.learn(
            total_timesteps=train_param['checkpoint_timesteps'],
            log_interval=500,
            reset_num_timesteps=False,
            callback=functools.partial(callback, test_env=test_env, interval=5000, n_episodes=20))

        print('\nSaving model {}.\n'.format(ckpt_file(ckpt_dir, ckpt_idx).name))
        model.save(str(ckpt_file(ckpt_dir, ckpt_idx)))
        ckpt_idx += 1

    print('Finished.')


def run_experiment(args, section_name=''):

    policy_fn = gnn_fwd.GnnFwd
    policy_param = {
        'num_processing_steps': json.loads(args.get('aggregation')),
    }

    env_name = args.get('env')

    def make_env():
        env = gym.make(env_name)
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=env.env.keys)
        return env

    env_param = {'make_env': make_env}
    test_env_param = {'make_env': make_env}

    train_param = {
        'use_checkpoint': args.getboolean('use_checkpoint'),
        'load_trained_policy': args.get('load_trained_policy'),
        'n_env': args.getint('n_env'),
        'n_steps': args.getint('n_steps'),
        'checkpoint_timesteps': args.getint('checkpoint_timesteps'),
        'total_timesteps': args.getint('total_timesteps'),
        'train_lr': args.getfloat('train_lr'),
        'cliprange': args.getfloat('cliprange'),
        'adam_epsilon': args.getfloat('adam_epsilon'),
    }

    if 'pretrain' in args and args.getboolean('pretrain'):
        pretrain_param = {
            'pretrain_dataset': args.get('pretrain_dataset'),
            'pretrain_epochs': args.getint('pretrain_epochs'),
            'pretrain_checkpoint_epochs': args.getint('pretrain_checkpoint_epochs'),
            'pretrain_batch': args.getint('pretrain_batch'),
            'pretrain_lr': args.getfloat('pretrain_lr'),
        }
    else:
        pretrain_param = None

    directory = Path('models/' + args.get('name') + section_name)

    train_helper(
        env_param=env_param,
        test_env_param=test_env_param,
        train_param=train_param,
        pretrain_param=pretrain_param,
        policy_fn=policy_fn,
        policy_param=policy_param,
        directory=directory)


def main():
    fname = sys.argv[1]
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)

    if config.sections():
        for section_name in config.sections():
            print(section_name)
            run_experiment(config[section_name], section_name)
    else:
        run_experiment(config[config.default_section])


if __name__ == '__main__':
    main()
