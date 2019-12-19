import tensorflow as tf
from graph_nets import graphs
from stable_baselines.common.policies import ActorCriticPolicy
import rl_comm.models as models
import numpy as np
from tensorflow.python.ops.ragged.ragged_util import repeat
from rl_comm.sparse_distribution import SparseMultiCategoricalProbabilityDistribution as SparseDistribution

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

            # can any of these help me?
            # https://github.com/deepmind/graph_nets/blob/f4ad83975276520e411befe5b9731057c6f57d7d/graph_nets/utils_tf.py

            # compute value
            self._value_fn = result_graph.globals
            self.q_value = None  # unused by PPO2

            n_agents = ac_space.nvec[0]
            n_robots = len(ac_space.nvec)
            n_edge = tf.reshape(n_edge, (batch_size,))
            tf.print(n_edge)

            # remap node indices back from batch to intra-graph indexes
            # remap_node_index = tf_repeat(, n_edge)
            cumsum = tf.reshape(tf.math.cumsum(n_node, exclusive=True), (batch_size,))
            remap_node_index = repeat(cumsum, n_edge, axis=0)

            orig_senders = result_graph.senders - remap_node_index
            orig_receivers = result_graph.receivers - remap_node_index

            # keep only edges in to controlled agents, out of uncontrolled agents
            mask = tf.logical_and(orig_receivers < n_robots, orig_senders >= n_robots)  # padded edges have sender = -1
            mask = tf.reshape(mask, (-1,))
            masked_senders = tf.boolean_mask(orig_senders, mask)
            masked_edges = tf.boolean_mask(result_graph.edges, mask, axis=0)

            # receivers2 needs to uniquely index the robots across all graphs
            print(tf.size(n_edge))
            cumsum2 = tf.reshape(tf.math.cumsum(tf.math.subtract(n_node, n_robots), exclusive=True), (batch_size,))
            # remap_node_index2 = tf_repeat(, n_edge)
            remap_node_index2 = repeat(cumsum2, n_edge, axis=0)
            receivers2 = tf.boolean_mask(result_graph.receivers - remap_node_index2, mask)

            indices = tf.reshape(tf.stack([receivers2, masked_senders], axis=1),
                                 (tf.reduce_sum(tf.cast(mask, tf.int32)), 2))
            indices = tf.cast(indices, tf.int64)  # why does this have to be
            dense_shape = tf.reshape(tf.cast([n_robots * batch_size, n_agents], tf.int64), (2,))
            masked_edges = tf.reshape(masked_edges, (-1,))

            # TODO use RaggedTensor instead
            # tf.nn.top_k - https://www.tensorflow.org/api_docs/python/tf/math/top_k?version=stable
            ragged_senders = tf.RaggedTensor.from_value_rowids(values=masked_senders, value_rowids=receivers2, nrows=n_robots)
            ragged_edges = tf.RaggedTensor.from_value_rowids(values=masked_edges, value_rowids=receivers2, nrows=n_robots)

            # Ragged logits?

            # TODO make a RaggedDistribution that takes

            logits = tf.sparse.SparseTensor(indices=indices, values=masked_edges, dense_shape=dense_shape)
            logits = tf.sparse.reorder(logits)

            dense_shape = tf.reshape(tf.cast([batch_size, n_robots * n_agents], tf.int64), (2,))
            self.sparse_logits = tf.cast(tf.sparse.reshape(logits, dense_shape), tf.float32)

            # TODO implement SparseMultiCategorical distribution - converting to dense is expensive
            # self.logits = tf.sparse.to_dense(self.sparse_logits, default_value=None)
            # self._policy = self.logits
            # self._proba_distribution = self.pdtype.proba_distribution_from_flat(self.logits)

            self._policy = self.sparse_logits
            self._proba_distribution = SparseDistribution(ac_space.nvec, self.sparse_logits)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})

        # # TODO cannot iterate, and fix based on sizes of policy and proba
        # neighbor_action = []
        # for (a, n) in zip(self.nodes, action):
        #     neighbor_action.append(n[a])

        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    @staticmethod
    def policy_param_string(p):
        """Return identifier string for policy parameter dict."""
        return 'gnnfwd'
