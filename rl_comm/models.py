from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt
from graph_nets import modules, blocks

from graph_nets import graphs
from stable_baselines.a2c.utils import ortho_init
from rl_comm.utils import segment_logsumexp, segment_softmax, segment_softmax_norm, segment_transformer
from graph_nets.blocks import unsorted_segment_max_or_zero
from graph_nets import utils_tf


class Identity(snt.AbstractModule):
    """Sonnet module implementing the identity."""
    def __init__(self, name="identity"):
        super(Identity, self).__init__(name=name)

    def _build(self, inputs):
        return tf.identity(inputs)


class AggregationNet(snt.AbstractModule):
    """
    Aggregation Net with learned aggregation filter
    """

    def __init__(self,
                 num_processing_steps=None,
                 latent_size=None,
                 n_layers=None,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 reducer=None,
                 out_init_scale=5.0,
                 name="AggregationNet"):
        super(AggregationNet, self).__init__(name=name)

        if num_processing_steps is None:
            self._proc_hops = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        else:
            self._proc_hops = num_processing_steps

        if reducer is None or reducer == 'max':
            reducer = unsorted_segment_max_or_zero
        elif reducer == 'logsumexp':
            reducer = segment_logsumexp
        elif reducer == 'transformer':
            reducer = segment_transformer
        elif reducer == 'softmax_norm':
            reducer = segment_softmax_norm
        elif reducer == 'mean':
            reducer = tf.math.unsorted_segment_mean
        elif reducer == 'sum':
            reducer = tf.math.unsorted_segment_sum
        else:
            raise ValueError('Unknown reducer!')

        if latent_size is None:
            latent_size = 16

        if n_layers is None:
            n_layers = 2

        self._num_processing_steps = len(self._proc_hops)
        self._n_stacked = latent_size * self._num_processing_steps

        def make_mlp():
            return snt.nets.MLP([latent_size] * n_layers, activate_final=True)

        core_a = blocks.EdgeBlock(
            edge_model_fn=lambda: Identity(),
            use_edges=False,
            use_receiver_nodes=False,
            use_sender_nodes=True,
            use_globals=False,
            name='LinearNodeAggGCN_core_a')

        # Then, edge data is aggregated onto the node.
        core_b = blocks.NodeBlock(
            node_model_fn=lambda: Identity(),
            use_received_edges=True,
            use_sent_edges=False,
            use_nodes=False,
            use_globals=False,
            received_edges_reducer=reducer,
            name='LinearNodeAggGCN_core_b')

        self._cores = [core_a, core_b]

        self._encoder = modules.GraphIndependent(make_mlp, make_mlp, make_mlp, name="encoder")
        self._decoder = modules.GraphIndependent(make_mlp, make_mlp, make_mlp, name="decoder")

        inits = {'w': ortho_init(out_init_scale), 'b': tf.constant_initializer(0.0)}

        # Transforms the outputs into the appropriate shapes.
        edge_fn = None if edge_output_size is None else lambda: snt.Linear(edge_output_size, initializers=inits,
                                                                           name="edge_output")
        node_fn = None if node_output_size is None else lambda: snt.Linear(node_output_size, initializers=inits,
                                                                           name="node_output")
        global_fn = None if global_output_size is None else lambda: snt.Linear(global_output_size,
                                                                               initializers=inits,
                                                                               name="global_output")
        with self._enter_variable_scope():
            self._output_transform = modules.GraphIndependent(edge_fn, node_fn, global_fn, name="output")

    def _build(self, input_op):
        latent = self._encoder(input_op)
        output_ops = [latent]
        for i in range(self._num_processing_steps):
            for j in range(self._proc_hops[i]):
                for c in self._cores:
                    latent = c(latent)

            decoded_op = self._decoder(latent)
            output_ops.append(decoded_op)
        return self._output_transform(utils_tf.concat(output_ops, axis=1))


class NonLinearGraphNet(snt.AbstractModule):
    """
    Aggregation Net with learned aggregation filter
    """

    def __init__(self,
                 num_processing_steps=None,
                 latent_size=None,
                 n_layers=None,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 reducer=None,
                 out_init_scale=5.0,
                 name="AggregationNet"):
        super(NonLinearGraphNet, self).__init__(name=name)

        if num_processing_steps is None:
            self._proc_hops = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        else:
            self._proc_hops = num_processing_steps

        if reducer is None or reducer == 'max':
            reducer = unsorted_segment_max_or_zero
        elif reducer == 'logsumexp':
            reducer = segment_logsumexp
        elif reducer == 'transformer':
            reducer = segment_transformer
        elif reducer == 'softmax_norm':
            reducer = segment_softmax_norm
        elif reducer == 'mean':
            reducer = tf.math.unsorted_segment_mean
        elif reducer == 'sum':
            reducer = tf.math.unsorted_segment_sum
        else:
            raise ValueError('Unknown reducer!')

        if latent_size is None:
            latent_size = 16

        if n_layers is None:
            n_layers = 2

        self._num_processing_steps = len(self._proc_hops)
        self._n_stacked = latent_size * self._num_processing_steps

        def make_mlp():
            return snt.nets.MLP([latent_size] * n_layers, activate_final=False)

        # def make_linear():
        #     return snt.Linear(latent_size)

        self._core = modules.GraphNetwork(
            edge_model_fn=make_mlp,
            node_model_fn=make_mlp,
            global_model_fn=make_mlp,
            edge_block_opt={'use_globals': False},
            node_block_opt={'use_globals': False, 'use_sent_edges': False},
            name="graph_net",
            reducer=reducer
        )

        self._encoder = modules.GraphIndependent(make_mlp, make_mlp, make_mlp, name="encoder")
        self._decoder = modules.GraphIndependent(make_mlp, make_mlp, make_mlp, name="decoder")

        inits = {'w': ortho_init(out_init_scale), 'b': tf.constant_initializer(0.0)}

        # Transforms the outputs into the appropriate shapes.
        edge_fn = None if edge_output_size is None else lambda: snt.Linear(edge_output_size, initializers=inits,
                                                                           name="edge_output")
        node_fn = None if node_output_size is None else lambda: snt.Linear(node_output_size, initializers=inits,
                                                                           name="node_output")
        global_fn = None if global_output_size is None else lambda: snt.Linear(global_output_size,
                                                                               initializers=inits,
                                                                               name="global_output")
        with self._enter_variable_scope():
            self._output_transform = modules.GraphIndependent(edge_fn, node_fn, global_fn, name="output")

    def _build(self, input_op):
        latent = self._encoder(input_op)
        output_ops = [latent]
        for i in range(self._num_processing_steps):
            for j in range(self._proc_hops[i]):
                latent = self._core(latent)

            decoded_op = self._decoder(latent)
            output_ops.append(decoded_op)
        return self._output_transform(utils_tf.concat(output_ops, axis=1))
