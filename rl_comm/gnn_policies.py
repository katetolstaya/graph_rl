import numpy as np

from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.policies import mlp_extractor
from stable_baselines.a2c.utils import linear

from gym_pdefense.envs import pdefense_env

import tensorflow as tf
import sonnet as snt
from graph_nets import graphs, modules, blocks




class MyMlpPolicy(ActorCriticPolicy):
    """

    Policy object that implements actor critic, using a a vanilla centralized
    MLP (2 layers of 64).

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
        net_arch=[dict(vf=[64,64], pi=[64,64])],
        w_agent=1,
        w_target=2,
        w_obs=2):

        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                scale=False)

        n_agents = ac_space.nvec.size
        n_targets = (np.prod(ob_space.shape) - n_agents**2 - n_agents*w_agent) // (n_agents + w_target + n_agents*w_obs)
        assert np.prod(ob_space.shape) == n_agents**2 + n_agents*w_agent + n_agents*n_targets + n_targets*w_target + n_agents*n_targets*w_obs, 'Broken game size computation.'

        (comm_adj, agent_node_data, obs_adj, target_node_data, obs_edge_data) = \
            pdefense_env.unpack_obs_graph_coord_tf(self.processed_obs, n_agents, n_targets, w_agent, w_target, w_obs)
        obs = tf.concat((tf.layers.flatten(agent_node_data), tf.layers.flatten(obs_edge_data)), axis=1)

        with tf.variable_scope("model", reuse=reuse):
            # Shared latent representation across entire team.
            pi_latent, vf_latent = mlp_extractor(obs, net_arch, tf.nn.relu)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

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




class OneNodePolicy(ActorCriticPolicy):
    """

    Policy object that implements actor critic, using the GraphNet API to
    reproduce the vanilla centralized MLP result (2 layers of 64)).

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
        net_arch=[dict(vf=[64,64], pi=[64,64])],
        w_agent=1,
        w_target=2,
        w_obs=2):

        super(OneNodePolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                scale=False)

        n_agents = ac_space.nvec.size
        n_targets = (np.prod(ob_space.shape) - n_agents**2 - n_agents*w_agent) // (n_agents + w_target + n_agents*w_obs)
        assert np.prod(ob_space.shape) == n_agents**2 + n_agents*w_agent + n_agents*n_targets + n_targets*w_target + n_agents*n_targets*w_obs, 'Broken game size computation.'

        (comm_adj, agent_node_data, obs_adj, target_node_data, obs_edge_data) = \
            pdefense_env.unpack_obs_graph_coord_tf(self.processed_obs, n_agents, n_targets, w_agent, w_target, w_obs)

        # Build observation graph. Concatenate all agent data and observation
        # data into a single node, and include no edges.
        B = tf.shape(obs_adj)[0]
        N = obs_adj.shape[1]
        nodes = tf.concat((tf.layers.flatten(agent_node_data), tf.layers.flatten(obs_edge_data)), axis=1)
        n_node = tf.fill((B,), 1)
        n_edge = tf.fill((B,), 0)
        in_graph = graphs.GraphsTuple(
            nodes=nodes,
            edges=None,
            globals=None,
            receivers=None,
            senders=None,
            n_node=n_node,
            n_edge=n_edge)

        with tf.variable_scope("model", reuse=reuse):

            # Transform the single node's data.
            state_mlp = blocks.NodeBlock(
                node_model_fn=lambda: snt.nets.MLP(tuple(net_arch[0]['pi']) + (N*2,), activate_final=False),
                use_received_edges=False,
                use_sent_edges=False,
                use_nodes=True,
                use_globals=False,
                name="pi_state_mlp")
            pi_g = state_mlp(in_graph)

            # Transform the single node's data.
            state_mlp = blocks.NodeBlock(
                node_model_fn=lambda: snt.nets.MLP(net_arch[0]['vf'], activate_final=True),
                use_received_edges=False,
                use_sent_edges=False,
                use_nodes=True,
                use_globals=False,
                name="vf_state_mlp")
            vf_g = state_mlp(in_graph)

            # Reduce to single global value.
            vf_state_agg = blocks.GlobalBlock(
                global_model_fn=lambda: snt.Linear(output_size=1),
                use_nodes=True,
                use_edges=False,
                use_globals=False,
                name='vf_state_agg')
            state_value_g = vf_state_agg(vf_g)

            # Reduce to per-agent action values. Not needed by A2C.
            vf_action_agg = blocks.NodeBlock(
                node_model_fn=lambda: snt.Linear(output_size=2),
                use_received_edges=False,
                use_sent_edges=False,
                use_nodes=True,
                use_globals=False,
                name='vf_action_agg')
            action_value_g = vf_action_agg(vf_g)

            # Team value.
            self._value_fn = state_value_g.globals
            self.q_value   = tf.reshape(action_value_g.nodes, (B, N*2))
            # Team policy.
            self._policy = tf.reshape(pi_g.nodes, (B, N*2))
            self._proba_distribution = self.pdtype.proba_distribution_from_flat(self._policy)

        self._setup_init()

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




class GnnCoord(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
        net_arch=[dict(vf=[64,64], pi=[64,64])],
        w_agent=1,
        w_target=2,
        w_obs=2,
        input_feat_layers=(64,64),
        feat_agg_layers=(64,64),
        pi_head_layers=(),
        vf_head_layers=()):

        super(GnnCoord, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                scale=False)
        n_agents = ac_space.nvec.size
        n_targets = (np.prod(ob_space.shape) - n_agents**2 - n_agents*w_agent) // (n_agents + w_target + n_agents*w_obs)
        assert np.prod(ob_space.shape) == n_agents**2 + n_agents*w_agent + n_agents*n_targets + n_targets*w_target + n_agents*n_targets*w_obs, 'Broken game size computation.'

        (comm_adj, agent_node_data, obs_adj, target_node_data, obs_edge_data) = \
            pdefense_env.unpack_obs_graph_coord_tf(self.processed_obs, n_agents, n_targets, w_agent, w_target, w_obs)

        # Build observation graph.
        B = tf.shape(obs_adj)[0]
        N = obs_adj.shape[1]
        M = obs_adj.shape[2]
        WO = obs_edge_data.shape[-1]
        WA = agent_node_data.shape[-1]

        # Identify dense edge data and receiver and node data.
        edges     = tf.reshape(obs_edge_data, (-1, WO))
        receivers = tf.reshape(tf.tile(tf.reshape(tf.range(N),(1,-1,1)), (B,1,M)), (-1,))
        nodes = tf.reshape(agent_node_data, (-1, WA))

        # Sparse edge data and receiver.
        edge_mask = tf.reshape(obs_adj, (-1,))
        edges     = tf.boolean_mask(edges,     edge_mask, axis=0)
        receivers = tf.boolean_mask(receivers, edge_mask)

        # Count nodes and edges.
        n_node = tf.fill((B,), obs_adj.shape[1])
        n_edge = tf.reduce_sum(tf.reduce_sum(obs_adj, -1), -1)

        obs_g = graphs.GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=None,
            receivers=receivers,
            senders=receivers, # irrelevant; arbitrary self-loops
            n_node=n_node,
            n_edge=n_edge)

        print_obs_g = tf.print(obs_g)

        with tf.variable_scope("model", reuse=reuse):

            # Independently transform all input features.
            model_fn = None
            if len(input_feat_layers) != 0:
                model_fn=lambda: snt.nets.MLP(input_feat_layers, activate_final=True)
            input_feat = modules.GraphIndependent(
                edge_model_fn=model_fn,
                node_model_fn=model_fn,
                name='input_feat'
                )

            # Aggregate local features.
            model_fn = None
            if len(feat_agg_layers) != 0:
                model_fn=lambda: snt.nets.MLP(feat_agg_layers, activate_final=True)
            else:
                model_fn=lambda: tf.identity
            feat_agg = blocks.NodeBlock(
                node_model_fn=model_fn,
                use_received_edges=True,
                use_sent_edges=False,
                use_nodes=True,
                use_globals=False,
                received_edges_reducer=tf.unsorted_segment_sum,
                name="feat_agg")

            # Local policy output.
            pi_mlp = modules.GraphIndependent(
                node_model_fn=lambda: snt.nets.MLP(pi_head_layers + (2,), activate_final=False),
                name='pi_mlp'
                )

            # Local latent value.
            model_fn = None
            if len(vf_head_layers) != 0:
                model_fn=lambda: snt.nets.MLP(vf_head_layers, activate_final=True)
            vf_latent_mlp = modules.GraphIndependent(
                node_model_fn=model_fn,
                name='vf_latent_mlp'
                )

            # Local action value output. Not needed by A2C.
            vf_action_mlp = modules.GraphIndependent(
                node_model_fn=lambda: snt.Linear(output_size=2),
                name='vf_action_mlp'
                )

            # Global state value output.
            vf_state_agg = blocks.GlobalBlock(
                global_model_fn=lambda: snt.Linear(output_size=1),
                use_nodes=True,
                use_edges=False,
                use_globals=False,
                name='vf_state_agg')

            latent_g = feat_agg(input_feat(obs_g))
            pi_g = pi_mlp(latent_g)
            vf_action_g = vf_action_mlp(vf_latent_mlp(latent_g))
            vf_state_g = vf_state_agg(vf_latent_mlp(latent_g))

            # Value.
            self._value_fn = vf_state_g.globals
            self.q_value   = tf.reshape(vf_action_g.nodes, (B, N*2))
            # Policy.
            with tf.control_dependencies([print_obs_g]):
                self._policy = tf.reshape(pi_g.nodes, (B, N*2))
                self._proba_distribution = self.pdtype.proba_distribution_from_flat(self._policy)

        self._setup_init()

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
