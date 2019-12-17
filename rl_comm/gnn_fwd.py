import tensorflow as tf
from graph_nets import graphs
from stable_baselines.common.policies import ActorCriticPolicy
import rl_comm.models as models


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

        (nodes, edges, senders, receivers) = self.processed_obs

        # node attributes are [agent_type: robot(1), target(0), visited: yes(1), no(0)]
        # edge attributes are [distance]

        agent_graph = graphs.GraphsTuple(
            nodes=nodes,
            edges=edges,
            globals=None,
            receivers=receivers,
            senders=senders,
            n_node=nodes.shape[0],
            n_edge=edges.shape[0])


        # https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html
        with tf.variable_scope("model", reuse=reuse):

            graph_model = models.EncodeProcessDecode(edge_output_size=1, global_output_size=1)
            result_graph = graph_model(agent_graph, num_processing_steps=num_processing_steps)

            # compute value and policy
            self._value_fn = result_graph.globals
            self._policy = result_graph.edges  # TODO convert to logits over edges of node


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
