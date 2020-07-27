# The MIT License
#
# Copyright (c) 2017 OpenAI (http://openai.com)
# Copyright (c) 2018-2019 Stable-Baselines Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import time
import gym
import os
import glob
import numpy as np
import tensorflow as tf
from collections import deque

# from tf_agents.replay_buffers.py_uniform_replay_buffer import PyUniformReplayBuffer
from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn, Runner
from rl_comm.utils import eval_env
from rl_comm.utils import ReplayBuffer


class PPO2(ActorCriticRLModel):
    """
    Proximal Policy Optimization algorithm (GPU version).
    Paper: https://arxiv.org/abs/1707.06347
    Code: https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ppo2/ppo2.py

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param cliprange_vf: (float or callable) Clipping parameter for the value function, it can be a function.
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling.
        To deactivate value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
                 adam_epsilon=1e-4, verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, lr_decay_factor=0.97,
                 lr_decay_steps=10000):

        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_steps = lr_decay_steps
        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.cliprange_vf = cliprange_vf
        self.edam_epsilon = adam_epsilon
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.clipfrac = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.value = None
        self.n_batch = None
        self.summary = None
        self._runner = None
        self.ep_info_buf = None
        self.episode_reward = None
        self.global_step = None
        self.trainer = None

        super().__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                         _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                         seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        if _init_setup_model:
            self.setup_model()

    def _make_runner(self):
        return Runner(env=self.env, model=self, n_steps=self.n_steps,
                      gamma=self.gamma, lam=self.lam)

    @property
    def runner(self) -> AbstractEnvRunner:
        if self._runner is None:
            self._runner = self._make_runner()
        return self._runner

    def _get_pretrain_placeholders(self):
        policy = self.act_model
        space = self.action_space
        if isinstance(space, gym.spaces.Discrete) or isinstance(space, gym.spaces.MultiDiscrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.n_batch = self.n_envs * self.n_steps

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, " \
                                                                 "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step, reuse=False, **self.policy_kwargs)
                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=True, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    # self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

                    vpred = train_model.value_flat

                    # Value function clipping: not present in the original PPO
                    if self.cliprange_vf is None:
                        # Default behavior (legacy from OpenAI baselines):
                        # use the same clipping as for the policy
                        self.clip_range_vf_ph = self.clip_range_ph
                        self.cliprange_vf = self.cliprange
                    elif isinstance(self.cliprange_vf, (float, int)) and self.cliprange_vf < 0:
                        # Original PPO implementation: no value function clipping
                        self.clip_range_vf_ph = None
                    else:
                        # Last possible behavior: clipping range
                        # specific to the value function
                        self.clip_range_vf_ph = tf.placeholder(tf.float32, [], name="clip_range_vf_ph")

                    if self.clip_range_vf_ph is None:
                        # No clipping
                        vpred_clipped = train_model.value_flat
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        vpred_clipped = self.old_vpred_ph + \
                                        tf.clip_by_value(train_model.value_flat - self.old_vpred_ph,
                                                         - self.clip_range_vf_ph, self.clip_range_vf_ph)

                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpred_clipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                      self.clip_range_ph), tf.float32))
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('loss', loss)

                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.summary.histogram(var.name, var)
                    grads = tf.gradients(loss, self.params)
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))

                self.global_step = tf.Variable(0, trainable=False)
                decayed_lr = tf.train.exponential_decay(self.learning_rate, self.global_step, self.lr_decay_steps,
                                                        self.lr_decay_factor)
                self.trainer = tf.train.AdamOptimizer(learning_rate=decayed_lr, epsilon=self.edam_epsilon)
                self._train = self.trainer.apply_gradients(grads, global_step=self.global_step)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.trainer._lr))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    if self.clip_range_vf_ph is not None:
                        tf.summary.scalar('clip_range_vf', tf.reduce_mean(self.clip_range_vf_ph))

                    tf.summary.scalar('old_neglog_action_probability', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.trainer._lr)
                        tf.summary.histogram('advantage', self.advs_ph)
                        tf.summary.histogram('clip_range', self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probability', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                self.train_model = train_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                self.summary = tf.summary.merge_all()

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update,
                    writer, states=None, cliprange_vf=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        :param cliprange_vf: (float) Clipping factor for the value function
        """
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions,
                  self.advs_ph: advs, self.rewards_ph: returns,
                  self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf

        if states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                curr_lr, curr_global_step, summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.trainer._lr, self.global_step, self.summary, self.pg_loss, self.vf_loss, self.entropy,
                     self.approxkl, self.clipfrac, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                curr_lr, curr_global_step, summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.trainer._lr, self.global_step, self.summary, self.pg_loss, self.vf_loss, self.entropy,
                     self.approxkl, self.clipfrac, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            curr_global_step, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.global_step, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def _setup_learn(self):
        """
        Check the environment.
        """
        if self.env is None:
            raise ValueError("Error: cannot train the model without a valid environment, please set an environment with"
                             "set_env(self, env) method.")
        if self.episode_reward is None:
            self.episode_reward = np.zeros((self.n_envs,))
        if self.ep_info_buf is None:
            self.ep_info_buf = deque(maxlen=100)

    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True):
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            t_first_start = time.time()

            n_updates = total_timesteps // self.n_batch
            for update in range(1, n_updates + 1):
                assert self.n_batch % self.nminibatches == 0, ("The number of minibatches (`nminibatches`) "
                                                               "is not a factor of the total number of samples "
                                                               "collected per rollout (`n_batch`), "
                                                               "some samples won't be used."
                                                               )
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = self.learning_rate(frac)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)
                # true_reward is the reward without discount
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = self.runner.run()
                self.num_timesteps += self.n_batch
                self.ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, batch_size):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_batch + epoch_num *
                                                                            self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, writer=writer,
                                                                 update=timestep, cliprange_vf=cliprange_vf_now))
                else:  # recurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_envs + epoch_num *
                                                                            self.n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, update=timestep,
                                                                 writer=writer, states=mb_states,
                                                                 cliprange_vf=cliprange_vf_now))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if writer is not None:
                    total_episode_reward_logger(self.episode_reward,
                                                true_reward.reshape((self.n_envs, self.n_steps)),
                                                masks.reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

            return self

    def pretrain(self, dataset, n_epochs=10, learning_rate=1e-4, ent_coef=0.0001,
                 adam_epsilon=1e-8, val_interval=None, test_env=None, ckpt_params=None, lr_decay_factor=0.97,
                 lr_decay_steps=5000):
        """
        Pretrain a model using behavior cloning:
        supervised learning given an expert dataset.

        NOTE: only Box and Discrete spaces are supported for now.

        :param ent_coef:
        :param ckpt_params:
        :param test_env: Test environment
        :param dataset: (ExpertDataset) Dataset manager
        :param n_epochs: (int) Number of iterations on the training set
        :param learning_rate: (float) Learning rate
        :param adam_epsilon: (float) the epsilon value for the adam optimizer
        :param val_interval: (int) Report training and validation losses every n epochs.
            By default, every 10th of the maximum number of epochs.
        :return: (BaseRLModel) the pretrained model
        """

        discrete_actions = isinstance(self.action_space, gym.spaces.Discrete)
        multidiscrete_actions = isinstance(self.action_space, gym.spaces.MultiDiscrete)

        assert discrete_actions or multidiscrete_actions, 'Only Discrete, MultiDiscrete action spaces are supported'
        if multidiscrete_actions:
            assert np.all(
                self.action_space.nvec == self.action_space.nvec[0]), "Ragged MultiDiscrete action spaces not allowed"
            n_actions = self.action_space.nvec[0]
            n_agents = len(self.action_space.nvec)

        # Validate the model every 10% of the total number of iteration
        if val_interval is None:
            # Prevent modulo by zero
            if n_epochs < 10:
                val_interval = 1
            else:
                val_interval = int(n_epochs / 10)

        tb_log_name = 'pretrain'
        writer = tf.summary.FileWriter(self.tensorboard_log + "/" + tb_log_name, flush_secs=30)
        # Do not save graph
        # writer.add_graph(self.graph)

        with self.graph.as_default():
            with tf.variable_scope('pretrain'):
                if multidiscrete_actions:
                    obs_ph, actions_ph, actions_logits_ph = self._get_pretrain_placeholders()
                    actions_ph = tf.reshape(actions_ph, (-1, n_agents))
                    one_hot_actions = tf.one_hot(actions_ph, n_actions)

                    actions_logits_ph = tf.reshape(actions_logits_ph, (-1, n_agents, n_actions))
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=actions_logits_ph,
                        labels=tf.stop_gradient(one_hot_actions),
                        axis=2
                    )
                    entropy_loss = tf.reduce_mean(self.act_model.proba_distribution.entropy())
                    loss = tf.reduce_mean(loss) - ent_coef * entropy_loss

                elif discrete_actions:
                    obs_ph, actions_ph, actions_logits_ph = self._get_pretrain_placeholders()
                    # actions_ph has a shape if (n_batch,), we reshape it to (n_batch, 1)
                    # so no additional changes is needed in the dataloader
                    actions_ph = tf.expand_dims(actions_ph, axis=1)
                    one_hot_actions = tf.one_hot(actions_ph, self.action_space.n)
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=actions_logits_ph,
                        labels=tf.stop_gradient(one_hot_actions)
                    )
                    entropy_loss = tf.reduce_mean(self.act_model.proba_distribution.entropy())
                    loss = tf.reduce_mean(loss) - ent_coef * entropy_loss

                else:
                    raise ValueError("Invalid action space")

                global_step = tf.Variable(0, trainable=False)
                decayed_lr = tf.train.exponential_decay(learning_rate, global_step, lr_decay_steps, lr_decay_factor)
                optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr, epsilon=adam_epsilon)
                optim_op = optimizer.minimize(loss, var_list=self.params, global_step=global_step)

                # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=adam_epsilon, decay=0.99)
                # optim_op = optimizer.minimize(loss, var_list=self.params)

            self.sess.run(tf.global_variables_initializer())

        if self.verbose > 0:
            print("Pretraining with Behavior Cloning...")

        if ckpt_params is not None:
            ckpt_idx = ckpt_params['ckpt_idx']
            ckpt_epochs = ckpt_params['ckpt_epochs']
            ckpt_file = ckpt_params['ckpt_file']
            ckpt_dir = ckpt_params['ckpt_dir']

        for epoch_idx in range(int(n_epochs)):
            train_loss = 0.0
            # Full pass on the training set
            for i in range(len(dataset.train_loader) - 1):
                expert_obs, expert_actions = dataset.get_next_batch('train')
                feed_dict = {
                    obs_ph: expert_obs,
                    actions_ph: expert_actions,
                }

                train_loss_, _ = self.sess.run([loss, optim_op], feed_dict)
                train_loss += train_loss_

                # if test_env is not None and i % 500 == 0:
                #     print('\nTesting...')
                #     results = eval_env(test_env, self, 10, render_mode='none')
                #     print('reward,          mean = {:.1f}, std = {:.1f}'.format(np.mean(results['reward']),
                #                                                                 np.std(results['reward'])))
                #     print()
            dataset.get_next_batch('train')
            train_loss /= (len(dataset.train_loader) - 1)

            if self.verbose > 0 and (epoch_idx + 1) % val_interval == 0:
                val_loss = 0.0
                # Full pass on the validation set
                for _ in range(len(dataset.val_loader) - 1):
                    expert_obs, expert_actions = dataset.get_next_batch('val')
                    val_loss_, = self.sess.run([loss], {obs_ph: expert_obs,
                                                        actions_ph: expert_actions})
                    val_loss += val_loss_
                dataset.get_next_batch('val')
                val_loss /= (len(dataset.val_loader) - 1)

                curr_lr, curr_global_step = self.sess.run([optimizer._lr, global_step])

                if self.verbose > 0:
                    print("==== Training progress {:.2f}% ====".format(100 * (epoch_idx + 1) / n_epochs))
                    print('Epoch {}'.format(epoch_idx + 1))
                    print("Training loss: {:.6f}, Validation loss: {:.6f}, Learning rate: {:10.3e}".format(train_loss,
                                                                                                           val_loss,
                                                                                                           curr_lr))
                    # print("Training loss: {:.6f}, Validation loss: {:.6f}".format(train_loss, val_loss))
                    print()

                    if writer is not None:
                        summary = tf.Summary(
                            value=[tf.Summary.Value(tag="pretrain_loss", simple_value=train_loss)])
                        writer.add_summary(summary, epoch_idx)

                        summary = tf.Summary(
                            value=[tf.Summary.Value(tag="pretrain_test_loss", simple_value=val_loss)])
                        writer.add_summary(summary, epoch_idx)

                if test_env is not None:
                    print('\nTesting...')
                    results = eval_env(test_env, self, 20, render_mode='none')
                    mean_reward = np.mean(results['reward'])
                    print('reward,          mean = {:.1f}, std = {:.1f}'.format(mean_reward,
                                                                                np.std(results['reward'])))
                    print()

                    if writer is not None:
                        summary = tf.Summary(
                            value=[tf.Summary.Value(tag="mean_reward", simple_value=mean_reward)])
                        writer.add_summary(summary, epoch_idx)
                        summary = tf.Summary(
                            value=[tf.Summary.Value(tag="learning_rate", simple_value=curr_lr)])
                        writer.add_summary(summary, epoch_idx)

            if ckpt_params is not None and epoch_idx % ckpt_epochs == 0:
                print('\nSaving model {}.\n'.format(ckpt_file(ckpt_dir, ckpt_idx).name))
                self.save(str(ckpt_file(ckpt_dir, ckpt_idx)))
                ckpt_idx += 1

            # Free memory
            del expert_obs, expert_actions
        if self.verbose > 0:
            print("Pretraining done.")
        return self

    def pretrain_dagger(self, env, n_epochs=10, learning_rate=1e-4, ent_coef=0.0001,
                        adam_epsilon=1e-8, buffer_size=1000, val_interval=None, test_env=None, ckpt_params=None,
                        batch_size=20, lr_decay_factor=0.97, lr_decay_steps=5000):
        """
        Pretrain a model using behavior cloning:
        supervised learning given an expert dataset.

        NOTE: only Box and Discrete spaces are supported for now.

        :param ent_coef:
        :param ckpt_params:
        :param test_env: Test environment
        :param env: Environment that implements a controller() method
        :param n_epochs: (int) Number of iterations on the training set
        :param learning_rate: (float) Learning rate
        :param adam_epsilon: (float) the epsilon value for the adam optimizer
        :param val_interval: (int) Report training and validation losses every n epochs.
            By default, every 10th of the maximum number of epochs.
        :return: (BaseRLModel) the pretrained model
        """
        discrete_actions = isinstance(self.action_space, gym.spaces.Discrete)
        multidiscrete_actions = isinstance(self.action_space, gym.spaces.MultiDiscrete)

        assert discrete_actions or multidiscrete_actions, 'Only Discrete, MultiDiscrete action spaces are supported'
        if multidiscrete_actions:
            assert np.all(
                self.action_space.nvec == self.action_space.nvec[0]), "Ragged MultiDiscrete action spaces not allowed"
            n_actions = self.action_space.nvec[0]
            n_agents = len(self.action_space.nvec)

        # Validate the model every 10% of the total number of iteration
        if val_interval is None:
            # Prevent modulo by zero
            if n_epochs < 10:
                val_interval = 1
            else:
                val_interval = int(n_epochs / 10)

        tb_log_name = 'pretrain'
        writer = tf.summary.FileWriter(self.tensorboard_log + "/" + tb_log_name, flush_secs=30)
        # Do not save graph
        # writer.add_graph(self.graph)

        with self.graph.as_default():
            with tf.variable_scope('pretrain'):
                if multidiscrete_actions:
                    obs_ph, actions_ph, actions_logits_ph = self._get_pretrain_placeholders()
                    actions_ph = tf.reshape(actions_ph, (-1, n_agents))
                    one_hot_actions = tf.one_hot(actions_ph, n_actions)

                    actions_logits_ph = tf.reshape(actions_logits_ph, (-1, n_agents, n_actions))
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=actions_logits_ph,
                        labels=tf.stop_gradient(one_hot_actions),
                        axis=2
                    )
                    entropy_loss = tf.reduce_mean(self.act_model.proba_distribution.entropy())
                    loss = tf.reduce_mean(loss) - ent_coef * entropy_loss

                elif discrete_actions:
                    obs_ph, actions_ph, actions_logits_ph = self._get_pretrain_placeholders()
                    # actions_ph has a shape if (n_batch,), we reshape it to (n_batch, 1)
                    # so no additional changes is needed in the dataloader
                    actions_ph = tf.expand_dims(actions_ph, axis=1)
                    one_hot_actions = tf.one_hot(actions_ph, self.action_space.n)
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=actions_logits_ph,
                        labels=tf.stop_gradient(one_hot_actions)
                    )
                    entropy_loss = tf.reduce_mean(self.act_model.proba_distribution.entropy())
                    loss = tf.reduce_mean(loss) - ent_coef * entropy_loss

                else:
                    raise ValueError("Invalid action space")

                global_step = tf.Variable(0, trainable=False)
                decayed_lr = tf.train.exponential_decay(learning_rate, global_step, lr_decay_steps, lr_decay_factor)
                optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr, epsilon=adam_epsilon)
                optim_op = optimizer.minimize(loss, var_list=self.params, global_step=global_step)

            self.sess.run(tf.global_variables_initializer())

        if self.verbose > 0:
            print("Pretraining with DAgger...")

        if ckpt_params is not None:
            ckpt_idx = ckpt_params['ckpt_idx']
            ckpt_epochs = ckpt_params['ckpt_epochs']
            ckpt_file = ckpt_params['ckpt_file']
            ckpt_dir = ckpt_params['ckpt_dir']

        buffer_size = 10000
        updates_per_step = 20
        n_train_episodes = 3000
        beta_coeff = 0.998
        # beta_coeff = 0.993

        beta = 1

        memory = ReplayBuffer(max_size=buffer_size)
        n_updates = 0
        epoch_idx = 0

        for i in range(n_train_episodes):

            beta = beta * beta_coeff  # max(beta * beta_coeff, 0.5)
            state = env.reset()
            done = False
            train_loss_ = 0
            train_reward = 0

            while not done:

                try:
                    optimal_action = env.env.env.controller(random=False, greedy=False, reset_solution=True)
                except AssertionError:
                    state = env.reset()
                    continue

                if np.random.binomial(1, beta) > 0:
                    action = optimal_action
                else:
                    state_arr = np.array(state).reshape((1, -1))
                    action, _ = self.predict(state_arr, deterministic=False)
                    action = np.array(action).reshape((-1, 1))

                next_state, reward, done, _ = env.step(action)
                train_reward += reward

                memory.insert((state, optimal_action))

                state = next_state

            if memory.curr_size > batch_size:
                for _ in range(updates_per_step):
                    samples = memory.sample(batch_size)
                    expert_obs, expert_actions = zip(*samples)

                    expert_obs_arr = np.concatenate(expert_obs, axis=0).reshape((batch_size, -1))
                    expert_actions_arr = np.concatenate(expert_actions, axis=0).reshape((batch_size, -1))

                    feed_dict = {
                        obs_ph: expert_obs_arr,
                        actions_ph: expert_actions_arr,
                    }

                    curr_lr, curr_global_step = self.sess.run([optimizer._lr, global_step])
                    train_loss_, _ = self.sess.run([loss, optim_op], feed_dict)

                    n_updates += 1
                epoch_idx += 1

            if self.verbose > 0 and (epoch_idx + 1) % val_interval == 0:
                if self.verbose > 0:
                    print("==== Training progress {:.2f}% ====".format(100 * (epoch_idx + 1) / n_epochs))
                    print('Epoch {}'.format(epoch_idx + 1))
                    print("Training loss: {:.6f}, Training reward: {:.6f}".format(train_loss_, train_reward))
                    print()

                    if writer is not None:
                        summary = tf.Summary(
                            value=[tf.Summary.Value(tag="pretrain_loss", simple_value=train_loss_)])
                        writer.add_summary(summary, epoch_idx)

                if test_env is not None:
                    print('\nTesting...')
                    results = eval_env(test_env, self, 20, render_mode='none')
                    mean_reward = np.mean(results['reward'])
                    print('reward,          mean = {:.1f}, std = {:.1f}'.format(mean_reward,
                                                                                np.std(results['reward'])))
                    print()

                    if writer is not None:
                        summary = tf.Summary(
                            value=[tf.Summary.Value(tag="mean_reward", simple_value=mean_reward)])
                        writer.add_summary(summary, epoch_idx)

                        summary = tf.Summary(
                            value=[tf.Summary.Value(tag="beta", simple_value=beta)])
                        writer.add_summary(summary, epoch_idx)

            if ckpt_params is not None and epoch_idx % ckpt_epochs == 0:
                print('\nSaving model {}.\n'.format(ckpt_file(ckpt_dir, ckpt_idx).name))
                self.save(str(ckpt_file(ckpt_dir, ckpt_idx)))
                ckpt_idx += 1

            # Free memory
            del expert_obs, expert_actions
        if self.verbose > 0:
            print("Pretraining done.")
        return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "cliprange_vf": self.cliprange_vf,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


class TensorboardWriter:
    def __init__(self, graph, tensorboard_log_path, tb_log_name, new_tb_log=True):
        """
        Create a Tensorboard writer for a code segment, and saves it to the log directory as its own run

        :param graph: (Tensorflow Graph) the model graph
        :param tensorboard_log_path: (str) the save path for the log (can be None for no logging)
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param new_tb_log: (bool) whether or not to create a new logging folder for tensorboard
        """
        self.graph = graph
        self.tensorboard_log_path = tensorboard_log_path
        self.tb_log_name = tb_log_name
        self.writer = None
        self.new_tb_log = new_tb_log

    def __enter__(self):
        if self.tensorboard_log_path is not None:
            latest_run_id = self._get_latest_run_id()
            if self.new_tb_log:
                latest_run_id = latest_run_id + 1
            save_path = os.path.join(self.tensorboard_log_path, "{}_{}".format(self.tb_log_name, latest_run_id))
            # self.writer = tf.summary.FileWriter(save_path, graph=self.graph)
            # do not save huge graph
            self.writer = tf.summary.FileWriter(save_path)
        return self.writer

    def _get_latest_run_id(self):
        """
        returns the latest run number for the given log name and log path,
        by finding the greatest number in the directories.

        :return: (int) latest run number
        """
        max_run_id = 0
        for path in glob.glob("{}/{}_[0-9]*".format(self.tensorboard_log_path, self.tb_log_name)):
            file_name = path.split(os.sep)[-1]
            ext = file_name.split("_")[-1]
            if self.tb_log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
                max_run_id = int(ext)
        return max_run_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            # self.writer.add_graph(self.graph)
            self.writer.flush()
