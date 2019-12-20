import tensorflow as tf
from graph_nets import graphs
from stable_baselines.common.policies import ActorCriticPolicy
import rl_comm.models as models
import numpy as np
from tensorflow.python.ops.array_ops import repeat_with_axis


def unpack_obs(obs):
    # TODO move this to the environment
    # these params are already in the env
    n_nodes = 925
    dim_nodes = 2
    max_edges = 5
    max_n_edges = n_nodes * max_edges
    dim_edges = 1

    # unpack node and edge data from flattened array
    shapes = ((n_nodes, dim_nodes), (max_n_edges, dim_edges), (max_n_edges, 1), (max_n_edges, 1))
    sizes = [np.prod(s) for s in shapes]
    tensors = tf.split(obs, sizes, axis=1)
    tensors = [tf.reshape(t, (-1,) + s) for (t, s) in zip(tensors, shapes)]
    nodes, edges, senders, receivers = tensors
    batch_size = tf.shape(nodes)[0]

    # TODO mask nodes too - assumes num. of landmarks is fixed (BAD)
    n_node = tf.fill((batch_size,), n_nodes)  # assume n nodes is fixed
    nodes = tf.reshape(nodes, (-1, dim_nodes))

    # compute edge mask and number of edges per graph
    mask = tf.reshape(tf.not_equal(senders, -1), (batch_size, -1))  # padded edges have sender = -1
    n_edge = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1)
    mask = tf.reshape(mask, (-1,))

    # flatten edge data
    edges = tf.reshape(edges, (-1, dim_edges))
    senders = tf.reshape(senders, (-1,))
    receivers = tf.reshape(receivers, (-1,))

    # mask edges
    edges = tf.boolean_mask(edges, mask, axis=0)
    senders = tf.boolean_mask(senders, mask)
    receivers = tf.boolean_mask(receivers, mask)

    # cast all indices to int
    n_node = tf.cast(n_node, tf.int32)
    n_edge = tf.cast(n_edge, tf.int32)
    senders = tf.cast(senders, tf.int32)
    receivers = tf.cast(receivers, tf.int32)

    # TODO this is a hack - want global outputs, but have no global inputs
    globs = tf.fill((batch_size, 1), 0.0)

    return batch_size, n_node, nodes, n_edge, edges, senders, receivers, globs


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
                 num_processing_steps=10):

        super(GnnFwd, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                     scale=False)

        batch_size, n_node, nodes, n_edge, edges, senders, receivers, globs = unpack_obs(self.processed_obs)

        agent_graph = graphs.GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=globs,
            receivers=receivers,
            senders=senders,
            n_node=n_node,
            n_edge=n_edge)

        # https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
        with tf.variable_scope("model", reuse=reuse):

            # TODO ensure that globals block shares weights for all nodes
            graph_model = models.EncodeProcessDecode(edge_output_size=1, global_output_size=1, node_output_size=8)
            result_graph = graph_model(agent_graph, num_processing_steps=num_processing_steps)
            result_graph = result_graph[-1]  # the last op is the decoded final processing step

            # compute value
            self._value_fn = result_graph.globals
            self.q_value = None  # unused by PPO2

            # keep only edges in to controlled agents and out of uncontrolled agents
            sender_type = tf.cast(tf.gather(nodes[:, 0], senders), tf.bool)
            receiver_type = tf.cast(tf.gather(nodes[:, 0], receivers), tf.bool)
            mask = tf.logical_and(tf.logical_not(sender_type), receiver_type)
            masked_edges = tf.boolean_mask(result_graph.edges, tf.reshape(mask, (-1,)), axis=0)

            # TODO assumed unchanged order of edges here - is this OK?
            n_actions = tf.cast(tf.reduce_sum(ac_space.nvec), tf.int32)
            self._policy = tf.reshape(masked_edges, (batch_size, n_actions))
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

    @staticmethod
    def policy_param_string(p):
        """Return identifier string for policy parameter dict."""
        return 'gnnfwd'
