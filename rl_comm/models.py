# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Model architectures for the demos."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import modules
from graph_nets.blocks import unsorted_segment_max_or_zero
from graph_nets import utils_tf
import sonnet as snt
import tensorflow as tf
from graph_nets import graphs
from stable_baselines.a2c.utils import ortho_init

# NUM_LAYERS = 2  # Hard-code number of layers in the edge/node/global models.
# LATENT_SIZE = 8  # Hard-code latent layer sizes for demos.

NUM_LAYERS = 2
LATENT_SIZE = 16


def make_mlp_model():
    """Instantiates a new MLP, followed by LayerNorm.

    The parameters of each new MLP are not shared with others generated by
    this function.

    Returns:
      A Sonnet module which contains the MLP and LayerNorm.
    """
    return snt.Sequential([
        snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True, activation=tf.tanh), snt.LayerNorm()
    ])


def make_linear_model():
    """Instantiates a new linear model.
    Returns:
      A Sonnet module which contains the linear layer.
    """
    return snt.nets.MLP([LATENT_SIZE], activate_final=False)


def make_mlp_model1():
    """Instantiates a new linear model.
    Returns:
      A Sonnet module which contains the linear layer.
    """
    return snt.Sequential([
        snt.nets.MLP([1], activate_final=False), snt.LayerNorm()
    ])


def make_linear_norm_model():
    """Instantiates a new linear model, followed by LayerNorm.
    Returns:
      A Sonnet module which contains the linear layer and LayerNorm.
    """
    return snt.Sequential([
        snt.nets.MLP([LATENT_SIZE], activate_final=False),
        snt.LayerNorm()
    ])


class AggregationNet(snt.AbstractModule):
    """
    Aggregation Net with learned aggregation filter
    """

    def __init__(self,
                 num_processing_steps,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 name="AggregationNet"):
        super(AggregationNet, self).__init__(name=name)

        # self._core = LinearGraphNetwork()

        self._aggregate = True
        self._use_globals = False if global_output_size is None else True
        # self._global_fn = None if global_output_size is None else make_mlp_model

        # self._core = MLPGraphNetwork(name="graph_net")

        # if not self._use_globals:
        #     graph_net_fn = make_linear_model
        # else:
        #     graph_net_fn = make_mlp_model

        # graph_net_fn = make_linear_norm_model

        graph_net_fn = make_linear_model

        self._core = modules.GraphNetwork(
            edge_model_fn=graph_net_fn,
            node_model_fn=graph_net_fn,
            global_model_fn=graph_net_fn,
            edge_block_opt={'use_receiver_nodes': False, 'use_globals': self._use_globals},
            node_block_opt={'use_globals': self._use_globals},
            name="graph_net"
            # ,reducer=unsorted_segment_max_or_zero
        )

        self._encoder = modules.GraphIndependent(make_mlp_model, make_mlp_model, make_mlp_model, name="encoder")
        self._decoder = modules.GraphIndependent(make_mlp_model, make_mlp_model, make_mlp_model, name="decoder")
        self._aggregation = modules.GraphIndependent(make_mlp_model, make_mlp_model, make_mlp_model, name="agg")

        self._num_processing_steps = num_processing_steps
        self._n_stacked = LATENT_SIZE * self._num_processing_steps

        edge_inits = {'w': ortho_init(1.0), 'b': tf.constant_initializer(0.0)}
        global_inits = {'w': ortho_init(1.0), 'b': tf.constant_initializer(0.0)}

        # Transforms the outputs into the appropriate shapes.
        edge_fn = None if edge_output_size is None else lambda: snt.Linear(edge_output_size, initializers=edge_inits,
                                                                           name="edge_output")
        node_fn = None if node_output_size is None else lambda: snt.Linear(node_output_size, initializers=edge_inits,
                                                                           name="node_output")
        global_fn = None if global_output_size is None else lambda: snt.Linear(global_output_size,
                                                                               initializers=global_inits,
                                                                               name="global_output")

        with self._enter_variable_scope():
            self._output_transform = modules.GraphIndependent(edge_fn, node_fn, global_fn, name="output")

    def _build(self, input_op):
        receivers = input_op.receivers
        senders = input_op.senders
        n_node = input_op.n_node
        n_edge = input_op.n_edge

        latent = self._encoder(input_op)
        latent0 = latent
        output_ops = []

        for _ in range(self._num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            decoded_op = self._decoder(latent)
            output_ops.append(decoded_op)

            # output_ops.append(latent)

        # if self._aggregate:

        stacked_edges = tf.stack([g.edges for g in output_ops], axis=1)
        stacked_nodes = tf.stack([g.nodes for g in output_ops], axis=1)
        stacked_globals = tf.stack([g.globals for g in output_ops], axis=1)

        # else:
        #     # out = self._output_transform(self._aggregation(output_ops[-1]))
        #     stacked_edges = tf.math.reduce_sum(tf.stack([g.edges for g in output_ops], axis=2), axis=2)
        #     stacked_nodes = tf.math.reduce_sum(tf.stack([g.nodes for g in output_ops], axis=2), axis=2)
        #     stacked_globals = tf.math.reduce_sum(tf.stack([g.globals for g in output_ops], axis=2), axis=2)

        stacked_globals = tf.reshape(stacked_globals, (-1, self._n_stacked))
        stacked_edges = tf.reshape(stacked_edges, (-1, self._n_stacked))
        stacked_nodes = tf.reshape(stacked_nodes, (-1, self._n_stacked))

        feature_graph = graphs.GraphsTuple(
            nodes=stacked_nodes,
            edges=stacked_edges,
            globals=stacked_globals,
            receivers=receivers,
            senders=senders,
            n_node=n_node,
            n_edge=n_edge)
        out = self._output_transform(self._aggregation(feature_graph))

        return out
