import tensorflow as tf
from graph_nets import graphs
from stable_baselines.common.policies import ActorCriticPolicy
import rl_comm.models as models
from gym_flock.envs.mapping_rad import MappingRadEnv


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
                 num_processing_steps=5):

        super(GnnFwd, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                     scale=False)

        batch_size, n_node, nodes, n_edge, edges, senders, receivers, globs = MappingRadEnv.unpack_obs(self.processed_obs)

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
            graph_model = models.EncodeProcessDecode(edge_output_size=1, global_output_size=1)
            result_graphs = graph_model(agent_graph, num_processing_steps=num_processing_steps)
            # compute value
            self._value_fn = result_graphs[-1].globals  #sum([g.globals for g in result_graphs]) #/ num_processing_steps
            edge_values = sum([g.edges for g in result_graphs]) #/ num_processing_steps

            # graph_model_policy = models.EncodeProcessDecode(edge_output_size=1)
            # graph_model_value = models.EncodeProcessDecode(global_output_size=1)
            # policy_graph = graph_model_policy(agent_graph, num_processing_steps=num_processing_steps)
            # value_graph = graph_model_value(agent_graph, num_processing_steps=num_processing_steps)
            # self._value_fn = value_graph[-1].globals
            # edge_values = sum([g.edges for g in policy_graph])

            self.q_value = None  # unused by PPO2

            # keep only edges in to controlled agents and out of uncontrolled agents
            sender_type = tf.cast(tf.gather(nodes[:, 0], senders), tf.bool)
            receiver_type = tf.cast(tf.gather(nodes[:, 0], receivers), tf.bool)
            mask = tf.logical_and(tf.logical_not(sender_type), receiver_type)
            masked_edges = tf.boolean_mask(edge_values, tf.reshape(mask, (-1,)), axis=0)

            # TODO assumed unchanged order of edges here - is this OK?
            n_actions = tf.cast(tf.reduce_sum(ac_space.nvec), tf.int32)
            self._policy = tf.reshape(masked_edges, (batch_size, n_actions))
            # temp = tf.Print(self._policy, [self._policy], summarize=-1)
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
