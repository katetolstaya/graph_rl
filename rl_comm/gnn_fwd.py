import tensorflow as tf
import math
from graph_nets import graphs
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
import rl_comm.models as models
from gym_flock.envs.spatial.coverage import CoverageEnv
from gym.spaces import MultiDiscrete
import numpy as np


class GnnFwd(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 num_processing_steps=None, latent_size=None, n_layers=None, reducer=None):

        super(GnnFwd, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                     scale=False)

        batch_size, n_node, nodes, n_edge, edges, senders, receivers, globs = CoverageEnv.unpack_obs(
            self.processed_obs, ob_space)

        agent_graph = graphs.GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=globs,
            receivers=receivers,
            senders=senders,
            n_node=n_node,
            n_edge=n_edge)

        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("value", reuse=reuse):
                self.value_model = models.LinearGraphNet(num_processing_steps=num_processing_steps,
                                                             latent_size=latent_size,
                                                             n_layers=n_layers, reducer=reducer,
                                                             node_output_size=1, name="value_model")
                value_graph = self.value_model(agent_graph)

                # sum the outputs of robot nodes to compute value
                node_type_mask = tf.reshape(tf.cast(nodes[:, 0], tf.bool), (-1,))
                # node_type_mask = tf.reshape(tf.reduce_any(tf.cast(nodes[:, 0:2], tf.bool), axis=1), (-1,))
                masked_nodes = tf.boolean_mask(value_graph.nodes, node_type_mask, axis=0)
                masked_nodes = tf.reshape(masked_nodes, (batch_size, len(ac_space.nvec)))
                self._value_fn = tf.reduce_sum(masked_nodes, axis=1, keepdims=True)

                # values = tf.reshape(value_graph.nodes, (batch_size, -1))
                # self._value_fn = tf.reduce_sum(values, axis=1, keepdims=True)

                self.q_value = None  # unused by PPO2

            with tf.variable_scope("policy", reuse=reuse):
                self.policy_model = models.LinearGraphNet(num_processing_steps=num_processing_steps,
                                                              latent_size=latent_size,
                                                              n_layers=n_layers, reducer=reducer,
                                                              edge_output_size=1, out_init_scale=1.0,
                                                              name="policy_model")
                policy_graph = self.policy_model(agent_graph)
                edge_values = policy_graph.edges

                # keep only edges for which senders are the landmarks, receivers are robots
                sender_type = tf.cast(tf.gather(nodes[:, 0], senders), tf.bool)
                receiver_type = tf.cast(tf.gather(nodes[:, 0], receivers), tf.bool)
                mask = tf.logical_and(tf.logical_not(sender_type), receiver_type)
                masked_edges = tf.boolean_mask(edge_values, tf.reshape(mask, (-1,)), axis=0)

                if isinstance(ac_space, MultiDiscrete):
                    n_actions = tf.cast(tf.reduce_sum(ac_space.nvec), tf.int32)
                else:
                    n_actions = tf.cast(ac_space.n, tf.int32)

                self._policy = tf.reshape(masked_edges, (batch_size, n_actions))
                self._proba_distribution = self.pdtype.proba_distribution_from_flat(self._policy)

        self._setup_init()

    def get_policy_model(self):
        return self.policy_model

    def get_value_model(self):
        return self.value_model

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})

        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    @staticmethod
    def policy_param_string(p):
        """Return identifier string for policy parameter dict."""
        return 'gnnfwd'


class MultiGnnFwd(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 num_processing_steps=None, latent_size=None, n_layers=None, reducer=None, n_gnn_layers=None):

        super(MultiGnnFwd, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, scale=False)

        batch_size, n_node, nodes, n_edge, edges, senders, receivers, globs = CoverageEnv.unpack_obs(
            self.processed_obs, ob_space)

        agent_graph = graphs.GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=globs,
            receivers=receivers,
            senders=senders,
            n_node=n_node,
            n_edge=n_edge)

        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("value", reuse=reuse):
                for i in range(n_gnn_layers - 1):
                    self.value_model_i = models.LinearGraphNet(num_processing_steps=num_processing_steps,
                                                                   latent_size=latent_size,
                                                                   n_layers=n_layers, reducer=reducer,
                                                                   node_output_size=latent_size,
                                                                   name="value_model" + str(i))
                    agent_graph = self.value_model_i(agent_graph)

                # The readout GNN layer
                self.value_model = models.LinearGraphNet(num_processing_steps=num_processing_steps,
                                                             latent_size=latent_size,
                                                             n_layers=n_layers, reducer=reducer,
                                                             node_output_size=1, name="value_model")
                value_graph = self.value_model(agent_graph)

                # sum the outputs of robot nodes to compute value
                node_type_mask = tf.reshape(tf.cast(nodes[:, 0], tf.bool), (-1,))
                # node_type_mask = tf.reshape(tf.reduce_any(tf.cast(nodes[:, 0:2], tf.bool), axis=1), (-1,))
                masked_nodes = tf.boolean_mask(value_graph.nodes, node_type_mask, axis=0)
                masked_nodes = tf.reshape(masked_nodes, (batch_size, len(ac_space.nvec)))
                self._value_fn = tf.reduce_sum(masked_nodes, axis=1, keepdims=True)
                # self._value_fn = tf.reduce_sum(value_graph.nodes, axis=1, keepdims=True)

                self.q_value = None  # unused by PPO2

            with tf.variable_scope("policy", reuse=reuse):
                for i in range(n_gnn_layers - 1):
                    self.policy_model_i = models.LinearGraphNet(num_processing_steps=num_processing_steps,
                                                                    latent_size=latent_size,
                                                                    n_layers=n_layers, reducer=reducer,
                                                                    node_output_size=latent_size,
                                                                    name="policy_model" + str(i))
                    agent_graph = self.policy_model_i(agent_graph)

                # The readout GNN layer
                self.policy_model = models.LinearGraphNet(num_processing_steps=num_processing_steps,
                                                              latent_size=latent_size,
                                                              n_layers=n_layers, reducer=reducer,
                                                              edge_output_size=1, out_init_scale=1.0,
                                                              name="policy_model")
                policy_graph = self.policy_model(agent_graph)
                edge_values = policy_graph.edges

                # keep only edges for which senders are the landmarks, receivers are robots
                sender_type = tf.cast(tf.gather(nodes[:, 0], senders), tf.bool)
                receiver_type = tf.cast(tf.gather(nodes[:, 0], receivers), tf.bool)
                mask = tf.logical_and(tf.logical_not(sender_type), receiver_type)
                masked_edges = tf.boolean_mask(edge_values, tf.reshape(mask, (-1,)), axis=0)

                if isinstance(ac_space, MultiDiscrete):
                    n_actions = tf.cast(tf.reduce_sum(ac_space.nvec), tf.int32)
                else:
                    n_actions = tf.cast(ac_space.n, tf.int32)

                self._policy = tf.reshape(masked_edges, (batch_size, n_actions))
                self._proba_distribution = self.pdtype.proba_distribution_from_flat(self._policy)

        self._setup_init()

    def get_policy_model(self):
        return self.policy_model

    def get_value_model(self):
        return self.value_model

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})

        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    @staticmethod
    def policy_param_string(p):
        """Return identifier string for policy parameter dict."""
        return 'gnnfwd'


class RecurrentGnnFwd(RecurrentActorCriticPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 num_processing_steps=None, latent_size=None, n_layers=None, reducer=None, state_shape=16):

        super(RecurrentGnnFwd, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                              state_shape=[CoverageEnv.get_number_nodes(ob_space) * state_shape * 2],
                                              scale=False)
        cur_state = self.states_ph

        value_fn = []
        policy = []
        for obs in tf.split(self.processed_obs, n_steps, 0):
            batch_size, n_node, nodes1, nodes2, n_edge, edges, senders, receivers, globs = CoverageEnv.unpack_obs_state(
                obs, ob_space, cur_state, state_shape)

            agent_graph1 = graphs.GraphsTuple(
                nodes=nodes1,
                edges=edges,
                globals=globs,
                receivers=receivers,
                senders=senders,
                n_node=n_node,
                n_edge=n_edge)

            agent_graph2 = graphs.GraphsTuple(
                nodes=nodes2,
                edges=edges,
                globals=globs,
                receivers=receivers,
                senders=senders,
                n_node=n_node,
                n_edge=n_edge)

            node_type_mask = tf.reshape(tf.cast(nodes1[:, 0], tf.bool), (-1,))
            node_type_mask2 = tf.reshape(tf.reduce_any(tf.cast(nodes1[:, 0:2], tf.bool), axis=1), (-1,))
            node_type_mask_float = tf.reshape(tf.cast(node_type_mask2, tf.float32), (-1, 1))

            with tf.variable_scope("model", reuse=reuse):
                with tf.variable_scope("value", reuse=reuse):
                    self.value_model = models.AggregationDiffNet(num_processing_steps=num_processing_steps,
                                                                 latent_size=latent_size,
                                                                 n_layers=n_layers, reducer=reducer,
                                                                 node_output_size=1 + state_shape, name="value_model")
                    value_graph = self.value_model(agent_graph2)

                    # sum the outputs of robot nodes to compute value
                    masked_nodes = tf.boolean_mask(value_graph.nodes[:, 0], node_type_mask, axis=0)
                    masked_nodes = tf.reshape(masked_nodes, (batch_size, len(ac_space.nvec)))
                    value_fn.append(tf.reduce_sum(masked_nodes, axis=1, keepdims=True))

                    # masked_nodes = tf.reshape(value_graph.nodes[:, 0], (-1, 1)) * node_type_mask_float
                    # masked_nodes = tf.reshape(masked_nodes, (batch_size, -1))
                    # value_fn.append(tf.reduce_sum(masked_nodes, axis=1, keepdims=True))

                    # state2 = value_graph.nodes[:, 1:]
                    state2 = value_graph.nodes[:, 1:] * node_type_mask_float

                with tf.variable_scope("policy", reuse=reuse):
                    self.policy_model = models.AggregationDiffNet(num_processing_steps=num_processing_steps,
                                                                  latent_size=latent_size,
                                                                  n_layers=n_layers, reducer=reducer,
                                                                  node_output_size=state_shape,
                                                                  edge_output_size=1, out_init_scale=1.0,
                                                                  name="policy_model")
                    policy_graph = self.policy_model(agent_graph1)
                    state1 = policy_graph.nodes * node_type_mask_float

                    edge_values = policy_graph.edges

                    # keep only edges for which senders are the landmarks, receivers are robots
                    sender_type = tf.cast(tf.gather(nodes2[:, 0], senders), tf.bool)
                    receiver_type = tf.cast(tf.gather(nodes2[:, 0], receivers), tf.bool)
                    mask = tf.logical_and(tf.logical_not(sender_type), receiver_type)
                    masked_edges = tf.boolean_mask(edge_values, tf.reshape(mask, (-1,)), axis=0)

                    if isinstance(ac_space, MultiDiscrete):
                        n_actions = tf.cast(tf.reduce_sum(ac_space.nvec), tf.int32)
                    else:
                        n_actions = tf.cast(ac_space.n, tf.int32)

                    policy.append(tf.reshape(masked_edges, (1, n_actions)))

                cur_state = tf.reshape(tf.concat([state1, state2], axis=1), tf.shape(self.states_ph))

        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("value", reuse=reuse):
                self._value_fn = tf.reshape(tf.stack(value_fn, axis=0), (n_steps, 1))
            with tf.variable_scope("policy", reuse=reuse):
                self._policy = tf.reshape(tf.stack(policy, axis=0), (n_steps, n_actions))
                self._proba_distribution = self.pdtype.proba_distribution_from_flat(self._policy)
        self.snew = cur_state
        self.q_value = None  # unused by PPO2

        self._setup_init()

    def get_policy_model(self):
        return self.policy_model

    def get_value_model(self):
        return self.value_model

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, snew, neglogp = self.sess.run(
                [self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                {self.obs_ph: obs, self.states_ph: state})
        else:
            action, value, snew, neglogp = self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                                         {self.obs_ph: obs, self.states_ph: state})
        return action, value, snew, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state})

    @staticmethod
    def policy_param_string(p):
        """Return identifier string for policy parameter dict."""
        return 'gnnfwd'


class MultiAgentGnnFwd(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
                 num_processing_steps=None, latent_size=None, n_layers=None, reducer=None, n_gnn_layers=None):

        super(MultiAgentGnnFwd, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                     scale=False)

        batch_size, n_node, nodes, n_edge, edges, senders, receivers, globs = CoverageEnv.unpack_obs(
            self.processed_obs, ob_space)

        n_robots = len(ac_space.nvec)

        value_models = []
        policy_models = []
        for i in range(n_gnn_layers-1):
            value_models_i = models.AggregationDiffNet(num_processing_steps=num_processing_steps,
                                                       latent_size=latent_size,
                                                       n_layers=n_layers, reducer=reducer,
                                                       node_output_size=latent_size,
                                                       name="value_model" + str(i))
            value_models.append(value_models_i)
            policy_model_i = models.AggregationDiffNet(num_processing_steps=num_processing_steps,
                                                       latent_size=latent_size,
                                                       n_layers=n_layers, reducer=reducer,
                                                       node_output_size=latent_size,
                                                       name="policy_model" + str(i))
            policy_models.append(policy_model_i)

        value_model = models.AggregationDiffNet(num_processing_steps=num_processing_steps,
                                                latent_size=latent_size,
                                                n_layers=n_layers, reducer=reducer,
                                                node_output_size=1, name="value_model")
        value_models.append(value_model)

        policy_model = models.AggregationDiffNet(num_processing_steps=num_processing_steps,
                                                 latent_size=latent_size,
                                                 n_layers=n_layers, reducer=reducer,
                                                 edge_output_size=1, out_init_scale=1.0,
                                                 name="policy_model")
        policy_models.append(policy_model)

        policies = []
        values = []
        robot_indices = tf.reshape(tf.math.cumsum(n_node, exclusive=True), (-1, 1))

        for n_agent in range(n_robots):

            nodes = nodes * tf.constant([[1.0, 1.0, 1.0, 0.0]])
            indices = tf.reshape(tf.stack([robot_indices + n_agent, 3 * tf.ones_like(robot_indices)], axis=1), (-1, 2))
            nodes = tf.tensor_scatter_nd_update(nodes, indices, tf.reshape(tf.ones_like(robot_indices, dtype=tf.float32), (-1,)))

            agent_graph = graphs.GraphsTuple(
                nodes=nodes,
                edges=edges,
                globals=globs,
                receivers=receivers,
                senders=senders,
                n_node=n_node,
                n_edge=n_edge)

            with tf.variable_scope("model", reuse=reuse):
                with tf.variable_scope("value", reuse=reuse):

                    value_graph = agent_graph
                    for model in value_models:
                        value_graph = model(value_graph)

                    # sum the outputs of robot nodes to compute value
                    node_type_mask = tf.reshape(tf.cast(nodes[:, -1], tf.bool), (-1,))
                    # node_type_mask = tf.reshape(tf.reduce_any(tf.cast(nodes[:, 0:2], tf.bool), axis=1), (-1,))
                    masked_nodes = tf.boolean_mask(value_graph.nodes, node_type_mask, axis=0)
                    masked_nodes = tf.reshape(masked_nodes, (batch_size, 1))
                    values.append(masked_nodes)
                    # self._value_fn = tf.reduce_sum(masked_nodes, axis=1, keepdims=True)

                with tf.variable_scope("policy", reuse=reuse):

                    policy_graph = agent_graph
                    for model in policy_models:
                        policy_graph = model(policy_graph)
                    edge_values = policy_graph.edges

                    # keep only edges for which senders are the landmarks, receivers are robots
                    sender_type = tf.cast(tf.gather(nodes[:, 0], senders), tf.bool)
                    receiver_type = tf.cast(tf.gather(nodes[:, -1], receivers), tf.bool)
                    mask = tf.logical_and(tf.logical_not(sender_type), receiver_type)
                    masked_edges = tf.boolean_mask(edge_values, tf.reshape(mask, (-1,)), axis=0)

                    if isinstance(ac_space, MultiDiscrete):
                        n_actions = tf.cast(ac_space.nvec[n_agent], tf.int32)
                    else:
                        n_actions = tf.cast(ac_space.n, tf.int32)

                    # self._policy = tf.reshape(masked_edges, (batch_size, n_actions))
                    policies.append(tf.reshape(masked_edges, (batch_size, n_actions)))

        self._value_fn = tf.reduce_sum(tf.concat(values, axis=1), axis=1, keepdims=True)
        self._policy = tf.concat(policies, axis=1)
        self._proba_distribution = self.pdtype.proba_distribution_from_flat(self._policy)
        self.q_value = None  # unused by PPO2

        self._setup_init()

    def get_policy_model(self):
        return self.policy_model

    def get_value_model(self):
        return self.value_model

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})

        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    @staticmethod
    def policy_param_string(p):
        """Return identifier string for policy parameter dict."""
        return 'gnnfwd'