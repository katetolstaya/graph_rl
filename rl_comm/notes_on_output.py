"""
Can define alternative output tensors to handle
* sampled action (stock normal training)
* noised, binarized deterministic action (stock normal testing) (should be worst)
Plus
* un-noised, un-binarized determinisitc action (should be best performing)
* un-noised, binarized deterministic action (real use case)


step() normally returns value and neglogp, but neither are needed for model.predict

argument 'deterministic' to predict() is passed to step(). Instead of only using
['False', 'True'], use ['False', 'noised', 'unnoised', 'binarized'].

"""


self._pdtype = make_proba_dist_type(ac_space)

# In our case, self._policy are output logits.
# These members are pre-defined for any ActorCriticPolicy
# For alternative outputs, can generate alternative deterministic outputs this way.
self._proba_distribution = self.pdtype.proba_distribution_from_flat(self._policy)
self._deterministic_action = self.proba_distribution.mode()
# Don't need the sampled output for anything other than training.
self._action = self.proba_distribution.sample()
self._neglogp = self.proba_distribution.neglogp(self.action)

@property
def action(self):
    """tf.Tensor: stochastic action, of shape (self.n_batch, ) + self.ac_space.shape."""
    return self._action

@property
def deterministic_action(self):
    """tf.Tensor: deterministic action, of shape (self.n_batch, ) + self.ac_space.shape."""
    return self._deterministic_action

@property
def proba_distribution(self):
    """ProbabilityDistribution: distribution of stochastic actions."""
    return self._proba_distribution




@property
def neglogp(self):
    """tf.Tensor: negative log likelihood of the action sampled by self.action."""
    return self._neglogp
