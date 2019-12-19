from stable_baselines.common.distributions import ProbabilityDistribution, ProbabilityDistributionType
from stable_baselines.a2c.utils import linear
import tensorflow as tf
import numpy as np


# class SparseMultiCategoricalProbabilityDistributionType(ProbabilityDistributionType):
#     def __init__(self, n_vec):
#         """
#         The probability distribution type for multiple categorical input
#
#         :param n_vec: ([int]) the vectors
#         """
#         # Cast the variable because tf does not allow uint32
#         self.n_vec = n_vec.astype(np.int32)
#         # Check that the cast was valid
#         assert (self.n_vec > 0).all(), "Casting uint32 to int32 was invalid"
#
#     def probability_distribution_class(self):
#         return SparseMultiCategoricalProbabilityDistribution
#
#     def proba_distribution_from_flat(self, flat):
#         return SparseMultiCategoricalProbabilityDistribution(self.n_vec, flat)
#
#     def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
#         pdparam = linear(pi_latent_vector, 'pi', sum(self.n_vec), init_scale=init_scale, init_bias=init_bias)
#         q_values = linear(vf_latent_vector, 'q', sum(self.n_vec), init_scale=init_scale, init_bias=init_bias)
#         return self.proba_distribution_from_flat(pdparam), pdparam, q_values
#
#     def param_shape(self):
#         return [sum(self.n_vec)]
#
#     def sample_shape(self):
#         return [len(self.n_vec)]
#
#     def sample_dtype(self):
#         return tf.int64


class SparseCategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, logits):
        """
        Probability distributions from categorical input

        :param logits: ([float]) the categorical logits input
        """
        self.logits = logits
        super(SparseCategoricalProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.logits

    def mode(self):
        return tf.argmax(self.logits, axis=-1)  # TODO

    def neglogp(self, x):
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])  # TODO
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=tf.stop_gradient(one_hot_actions))

    def kl(self, other):
        a_0 = self.logits - tf.sparse.reduce_max(self.logits, axis=-1, keepdims=True)
        a_1 = other.logits - tf.sparse.reduce_max(other.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        exp_a_1 = tf.exp(a_1)
        z_0 = tf.sparse.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        z_1 = tf.sparse.reduce_sum(exp_a_1, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.sparse.reduce_sum(p_0 * (a_0 - tf.log(z_0) - a_1 + tf.log(z_1)), axis=-1)

    def entropy(self):
        a_0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        exp_a_0 = tf.exp(a_0)
        z_0 = tf.sparse.reduce_sum(exp_a_0, axis=-1, keepdims=True)
        p_0 = exp_a_0 / z_0
        return tf.sparse.reduce_sum(p_0 * (tf.log(z_0) - a_0), axis=-1)

    def sample(self):
        # Gumbel-max trick to sample
        # a categorical distribution (see http://amid.fish/humble-gumbel)

        # TODO - add noise only to nonzero elements
        uniform = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)

        # TODO argmax over non-zero only
        return tf.argmax(self.logits - tf.log(-tf.log(uniform)), axis=-1)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the categorical logits input
        :return: (ProbabilityDistribution) the instance from the given categorical input
        """
        return cls(flat)


class SparseMultiCategoricalProbabilityDistribution(ProbabilityDistribution):
    def __init__(self, nvec, flat):
        """
        Probability distributions from multicategorical input

        :param nvec: ([int]) the sizes of the different categorical inputs
        :param flat: ([float]) the categorical logits input
        """
        self.flat = flat
        self.categoricals = list(
            map(SparseCategoricalProbabilityDistribution, tf.sparse.split(sp_input=flat, num_split=len(nvec), axis=-1)))
        super(SparseMultiCategoricalProbabilityDistribution, self).__init__()

    def flatparam(self):
        return self.flat

    def mode(self):
        return tf.stack([p.mode() for p in self.categoricals], axis=-1)

    def neglogp(self, x):
        return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x, axis=-1))])

    def kl(self, other):
        return tf.add_n([p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])

    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])

    def sample(self):
        return tf.stack([p.sample() for p in self.categoricals], axis=-1)

    @classmethod
    def fromflat(cls, flat):
        """
        Create an instance of this from new logits values

        :param flat: ([float]) the multi categorical logits input
        :return: (ProbabilityDistribution) the instance from the given multi categorical input
        """
        raise NotImplementedError
