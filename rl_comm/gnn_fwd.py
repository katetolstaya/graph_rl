import numpy as np
import sonnet as snt
import tensorflow as tf

from graph_nets import graphs, modules, blocks
from stable_baselines.common.policies import ActorCriticPolicy

from gym_pdefense.envs import pdefense_env

def mlp_model_fn(layers, default, activate_final):
    """
    Return model_fn for mlp, or default if len(layers) == 0. Typical
    defaults are None or lambda: tf.identity.
    """
    if len(layers) != 0:
        model_fn=lambda: snt.nets.MLP(layers, activate_final=activate_final)
    else:
        model_fn=default
    return model_fn

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
        msg_enc_layers=(),
        msg_size=8,
        msg_dec_layers=(),
        msg_agg_layers=(),
        pi_head_layers=(),
        vf_local_head_layers=(),
        vf_global_head_layers=()):

        super(GnnFwd, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
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
        obs_edges     = tf.reshape(obs_edge_data, (-1, WO))
        obs_receivers = tf.reshape( # receiver index of entry obs_adj[b][n][m] is N*b + n.
                        tf.tile(tf.reshape(N*tf.range(B),(-1,1,1)), (1,N,M)) + \
                        tf.tile(tf.reshape(tf.range(N),(1,-1,1)), (B,1,M)),
                        (-1,)
                    )
        nodes = tf.reshape(agent_node_data, (-1, WA))

        # Sparse edge data and receiver.
        obs_edge_mask = tf.reshape(obs_adj, (-1,))
        obs_edges     = tf.boolean_mask(obs_edges,     obs_edge_mask, axis=0)
        obs_receivers = tf.boolean_mask(obs_receivers, obs_edge_mask)

        # Count nodes and obs_edges.
        n_node = tf.fill((B,), N)
        n_edge = tf.reduce_sum(tf.reduce_sum(obs_adj, -1), -1)

        obs_g = graphs.GraphsTuple(
            nodes=nodes,
            edges=obs_edges,
            globals=None,
            receivers=obs_receivers,
            senders=obs_receivers, # irrelevant; arbitrary self-loops
            n_node=n_node,
            n_edge=n_edge)

        # Build communication edges over same nodes.
        comm_receivers = tf.reshape( # receiver index of entry comm_adj[b][n_rx][n_tx] is N*b + n_rx.
                        tf.tile(tf.reshape(N*tf.range(B),(-1,1,1)), (1,N,N)) + \
                        tf.tile(tf.reshape(tf.range(N),(1,-1,1)), (B,1,N)),
                        (-1,)
                    )
        comm_senders = tf.reshape( # sender index of entry comm_adj[b][n_rx][n_tx] is N*b + n_tx.
                        tf.tile(tf.reshape(N*tf.range(B),(-1,1,1)), (1,N,N)) + \
                        tf.tile(tf.reshape(tf.range(N),(1,1,-1)), (B,N,1)),
                        (-1,)
                    )

        # Sparse edge data and receiver.
        comm_edge_mask = tf.reshape(comm_adj, (-1,))
        comm_receivers = tf.boolean_mask(comm_receivers, comm_edge_mask)
        comm_senders   = tf.boolean_mask(comm_senders,   comm_edge_mask)

        # Count nodes and comm_edges.
        comm_n_edge = tf.reduce_sum(tf.reduce_sum(comm_adj, -1), -1)

        print_obs_g = tf.print(obs_g)

        with tf.variable_scope("model", reuse=reuse):

            # Independently transform all input features.
            input_feat = modules.GraphIndependent(
                edge_model_fn=mlp_model_fn(input_feat_layers, default=None, activate_final=True),
                node_model_fn=mlp_model_fn(input_feat_layers, default=None, activate_final=True),
                name='input_feat'
                )

            # Aggregate local features.
            feat_agg = blocks.NodeBlock(
                node_model_fn=mlp_model_fn(feat_agg_layers, default=lambda: tf.identity, activate_final=True),
                use_received_edges=True,
                use_sent_edges=False,
                use_nodes=True,
                use_globals=False,
                received_edges_reducer=tf.unsorted_segment_sum,
                name="feat_agg")

            # Encode and broadcast messages.
            msg_enc = blocks.EdgeBlock(
                edge_model_fn=lambda: snt.nets.MLP(msg_enc_layers + (msg_size,), activate_final=False),
                use_edges=False,
                use_receiver_nodes=False,
                use_sender_nodes=True,
                use_globals=False,
                name='msg_enc')

            # Decode and aggregate messages.
            msg_dec = modules.GraphIndependent(
                edge_model_fn=mlp_model_fn(msg_dec_layers, default=None, activate_final=True),
                name='msg_dec')
            msg_agg = blocks.NodeBlock(
                node_model_fn=mlp_model_fn(msg_agg_layers, default=lambda: tf.identity, activate_final=True),
                use_received_edges=True,
                use_sent_edges=False,
                use_nodes=True,
                use_globals=False,
                received_edges_reducer=tf.unsorted_segment_sum,
                name="msg_agg")

            # Local policy output.
            pi_mlp = modules.GraphIndependent(
                node_model_fn=lambda: snt.nets.MLP(pi_head_layers + (2,), activate_final=False),
                name='pi_mlp')

            # Local latent value.
            vf_latent_mlp = modules.GraphIndependent(
                node_model_fn=mlp_model_fn(vf_local_head_layers, default=None, activate_final=True),
                name='vf_latent_mlp')

            # Local action value output. Not needed by A2C.
            vf_action_mlp = modules.GraphIndependent(
                node_model_fn=lambda: snt.Linear(output_size=2),
                name='vf_action_mlp')

            # Global state value output.
            vf_state_agg = blocks.GlobalBlock(
                global_model_fn=lambda: snt.nets.MLP(vf_global_head_layers + (1,), activate_final=False),
                use_nodes=True,
                use_edges=False,
                use_globals=False,
                name='vf_state_agg')

            # Compute latent features based on observability graph.
            latent_g = feat_agg(input_feat(obs_g))

            # Exchange information over communication graph.
            latent_g = latent_g.replace(
                edges=None, senders=comm_senders, receivers=comm_receivers, n_edge=comm_n_edge)
            msg_g = msg_enc(latent_g)
            print_msg_g = tf.print(tf.shape(msg_g.edges))
            latent_g = msg_agg(msg_dec(msg_g))

            # Compute policy and value.
            pi_g = pi_mlp(latent_g)
            vf_action_g = vf_action_mlp(vf_latent_mlp(latent_g))
            vf_state_g = vf_state_agg(vf_latent_mlp(latent_g))

            # Value.
            self._value_fn = vf_state_g.globals
            self.q_value   = tf.reshape(vf_action_g.nodes, (B, N*2))
            # Policy.
            # with tf.control_dependencies([print_msg_g]):
            with tf.control_dependencies([]):
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
