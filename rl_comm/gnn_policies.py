from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.policies import mlp_extractor
from stable_baselines.a2c.utils import linear

from gym_pdefense.envs import pdefense_env

import tensorflow as tf
import sonnet as snt
from graph_nets import graphs, modules, blocks

# EDGE_SIZE = 16
# NODE_SIZE = 16

LATENT_SIZE = 32
NUM_LAYERS = 1

class MyMlpPolicy(ActorCriticPolicy):
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
    :param act_fun: (tf.func) the activation function to use in the neural network.
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
        net_arch=[dict(vf=[64,64], pi=[64,64])], act_fun=tf.tanh):

        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                scale=False)

        with tf.variable_scope("model", reuse=reuse):
            pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun)

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
    :param act_fun: (tf.func) the activation function to use in the neural network.
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False,
        net_arch=[dict(vf=[64,64], pi=[64,64])], act_fun=tf.tanh):

        super(GnnCoord, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                                scale=False)

        n_agents      = 2
        n_targets     = 2
        w_agent_data  = 1
        w_target_data = 2
        w_obs_data    = 2

        (comm_adj, agent_node_data, obs_adj, target_node_data, obs_edge_data) = \
            pdefense_env.unpack_obs_graph_coord_tf(self.processed_obs, n_agents, n_targets, w_agent_data, w_target_data, w_obs_data)

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

        obs_graph = graphs.GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=None,
            receivers=receivers,
            senders=receivers, # irrelevant; arbitrary self-loops
            n_node=n_node,
            n_edge=n_edge)

        # Transform each observation edge data.
        obs_mlp = blocks.EdgeBlock(
            # edge_model_fn=lambda: snt.Linear(output_size=EDGE_SIZE),
            edge_model_fn=lambda: snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True),
            use_edges=True,
            use_receiver_nodes=False,
            use_sender_nodes=False,
            use_globals=False,
            name="obs_mlp")
        # Transform each agent state node data.
        state_mlp = blocks.NodeBlock(
            # node_model_fn=lambda: snt.Linear(output_size=NODE_SIZE),
            node_model_fn=lambda: snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True),
            use_received_edges=False,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=False,
            name="state_mlp")
        # Reduce observations, concatenate with agent state, and apply mlp.
        agent_agg = blocks.NodeBlock(
            # node_model_fn=lambda: snt.Linear(output_size=NODE_SIZE),
            node_model_fn=lambda: snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True),
            use_received_edges=True,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=False,
            received_edges_reducer=tf.unsorted_segment_sum,
            name="agent_mlp")

        pi_g = agent_agg(state_mlp(obs_mlp(obs_graph)))

        # Transform each observation edge data.
        obs_mlp = blocks.EdgeBlock(
            # edge_model_fn=lambda: snt.Linear(output_size=EDGE_SIZE),
            edge_model_fn=lambda: snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True),
            use_edges=True,
            use_receiver_nodes=False,
            use_sender_nodes=False,
            use_globals=False,
            name="obs_mlp")
        # Transform each agent state node data.
        state_mlp = blocks.NodeBlock(
            # node_model_fn=lambda: snt.Linear(output_size=NODE_SIZE),
            node_model_fn=lambda: snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True),
            use_received_edges=False,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=False,
            name="state_mlp")
        # Reduce observations, concatenate with agent state, and apply mlp.
        agent_agg = blocks.NodeBlock(
            # node_model_fn=lambda: snt.Linear(output_size=NODE_SIZE),
            node_model_fn=lambda: snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True),
            use_received_edges=True,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=False,
            received_edges_reducer=tf.unsorted_segment_sum,
            name="agent_mlp")

        vf_g = agent_agg(state_mlp(obs_mlp(obs_graph)))



        # g = obs_network(obs_graph)

        # graph_network = modules.GraphNetwork(
        #     edge_model_fn=lambda: snt.Linear(output_size=EDGE_SIZE))

        with tf.variable_scope("model", reuse=reuse):
            # pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(agent_node_data), net_arch, act_fun)

            pi_latent = tf.reshape(pi_g.nodes, (B,N*LATENT_SIZE))
            vf_latent = tf.reshape(vf_g.nodes, (B,N*LATENT_SIZE))

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
