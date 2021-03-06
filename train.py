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
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize
from rl_comm.dataset import ExpertDataset

from rl_comm.gnn_fwd import GnnFwd, RecurrentGnnFwd, MultiGnnFwd, MultiAgentGnnFwd
from rl_comm.ppo2 import PPO2
from rl_comm.utils import ckpt_file, callback


def train_helper(env_param, test_env_param, train_param, pretrain_param, policy_fn, policy_param, directory, env=None, test_env=None):
    save_dir = Path(directory)
    tb_dir = save_dir / 'tb'
    ckpt_dir = save_dir / 'ckpt'
    for d in [save_dir, tb_dir, ckpt_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if env is None:
        if 'normalize_reward' in train_param and train_param['normalize_reward']:
            env = VecNormalize(env, norm_obs=False, norm_reward=True)
        else:
            env = SubprocVecEnv([env_param['make_env']] * train_param['n_env'])

    if test_env is None:
        test_env = SubprocVecEnv([test_env_param['make_env']])

    if train_param['use_checkpoint']:
        # Find latest checkpoint index.
        ckpt_list = sorted(glob.glob(str(ckpt_dir) + '/*.pkl'))
        if len(ckpt_list) == 0:
            ckpt_idx = None
        else:
            ckpt_idx = int(ckpt_list[-2][-7:-4])
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
            ent_coef=train_param['ent_coef'],
            vf_coef=train_param['vf_coef'],
            verbose=1,
            tensorboard_log=str(tb_dir),
            full_tensorboard_log=False,
            lr_decay_factor=train_param['lr_decay_factor'],
            lr_decay_steps=train_param['lr_decay_steps'],
        )

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

        if len(pretrain_param['pretrain_dataset']) > 0:
            dataset = ExpertDataset(expert_path=pretrain_param['pretrain_dataset'], traj_limitation=200,
                                    batch_size=pretrain_param['pretrain_batch'], randomize=True)

            model.pretrain(dataset, n_epochs=pretrain_param['pretrain_epochs'],
                           learning_rate=pretrain_param['pretrain_lr'],
                           val_interval=1, test_env=test_env, ckpt_params=ckpt_params,
                           ent_coef=pretrain_param['pretrain_ent_coef'],
                           lr_decay_factor=pretrain_param['pretrain_lr_decay_factor'],
                           lr_decay_steps=pretrain_param['pretrain_lr_decay_steps'])

            del dataset
            ckpt_idx += int(pretrain_param['pretrain_epochs'] / ckpt_params['ckpt_epochs'])
        else:
            model.pretrain_dagger(env_param['make_env'](), n_epochs=pretrain_param['pretrain_epochs'],
                                  learning_rate=pretrain_param['pretrain_lr'],
                                  val_interval=pretrain_param['pretrain_checkpoint_epochs'], test_env=test_env,
                                  ckpt_params=ckpt_params,
                                  ent_coef=pretrain_param['pretrain_ent_coef'],
                                  batch_size=pretrain_param['pretrain_batch'],
                                  lr_decay_factor=pretrain_param['pretrain_lr_decay_factor'],
                                  lr_decay_steps=pretrain_param['pretrain_lr_decay_steps'])

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
    # env.close()
    # test_env.close()
    del model

    return env, test_env


def run_experiment(args, section_name='', env=None, test_env=None):

    policy_param = {
        'num_processing_steps': json.loads(args.get('aggregation', '[1,1,1,1,1,1,1,1,1,1]')),
        'latent_size': args.getint('latent_size', 16),
        'n_layers': args.getint('n_layers', 3),
        'reducer': args.get('reducer', 'mean'),
        'model_type': args.get('model_type', 'identity'),
        'n_node_feat': args.getint('n_node_feat', 3)
    }
    policy_type = args.get('policy', 'GNNFwd')

    if policy_type == 'GNNFwd':
        policy_fn = GnnFwd
    elif policy_type == 'MultiGNNFwd':
        policy_fn = MultiGnnFwd
        policy_param['n_gnn_layers'] = args.getint('n_gnn_layers', 1)
    elif policy_type == 'RecurrentGNNFwd':
        policy_fn = RecurrentGnnFwd
        policy_param['state_shape'] = args.getint('rnn_state_shape', 16)
    elif policy_type == 'MultiAgentGNNFwd':
        policy_fn = MultiAgentGnnFwd
        policy_param['n_gnn_layers'] = args.getint('n_gnn_layers', 1)
    else:
        raise ValueError('Unknown policy type.')

    env_name = args.get('env', 'CoverageARL-v0')

    def make_env():
        env = gym.make(env_name)
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=env.env.keys)
        return env

    env_param = {'make_env': make_env}
    test_env_param = {'make_env': make_env}

    train_param = {
        'use_checkpoint': args.getboolean('use_checkpoint', False),
        'load_trained_policy': args.get('load_trained_policy', ''),
        'normalize_reward': args.get('normalize_reward', False),
        'n_env': args.getint('n_env', 4),
        'n_steps': args.getint('n_steps', 10),
        'checkpoint_timesteps': args.getint('checkpoint_timesteps', 10000),
        'total_timesteps': args.getint('total_timesteps', 50000000),
        'train_lr': args.getfloat('train_lr', 1e-4),
        'cliprange': args.getfloat('cliprange', 0.2),
        'adam_epsilon': args.getfloat('adam_epsilon', 1e-6),
        'vf_coef': args.getfloat('vf_coef', 0.5),
        'ent_coef': args.getfloat('ent_coef', 0.01),
        'lr_decay_factor': args.getfloat('lr_decay_factor', 0.97),
        'lr_decay_steps': args.getfloat('lr_decay_steps', 10000),
    }

    if 'pretrain' in args and args.getboolean('pretrain'):
        pretrain_param = {
            'pretrain_dataset': args.get('pretrain_dataset'),
            'pretrain_epochs': args.getint('pretrain_epochs', 100),
            'pretrain_checkpoint_epochs': args.getint('pretrain_checkpoint_epochs', 2),
            'pretrain_batch': args.getint('pretrain_batch', 32),
            'pretrain_lr': args.getfloat('pretrain_lr', 1e-3),
            'pretrain_ent_coef': args.getfloat('pretrain_ent_coef', 1e-6),
            'pretrain_lr_decay_factor': args.getfloat('pretrain_lr_decay_factor', 0.95),
            'pretrain_lr_decay_steps': args.getfloat('pretrain_lr_decay_steps', 200),
        }
    else:
        pretrain_param = None

    directory = Path('models/' + args.get('name') + section_name)

    env, test_env = train_helper(
        env_param=env_param,
        test_env_param=test_env_param,
        train_param=train_param,
        pretrain_param=pretrain_param,
        policy_fn=policy_fn,
        policy_param=policy_param,
        directory=directory,
        env=env, test_env=test_env)
    return env, test_env


def main():
    fname = sys.argv[1]
    config_file = path.join(path.dirname(__file__), fname)
    config = configparser.ConfigParser()
    config.read(config_file)
    if config.sections():
        env = None
        test_env = None
        for section_name in config.sections():
            print(section_name)
            env, test_env = run_experiment(config[section_name], section_name, env, test_env)
    else:
        run_experiment(config[config.default_section])


if __name__ == '__main__':
    main()
