from __future__ import absolute_import


import tensorflow as tf
slim = tf.contrib.slim
import squeezenext_architecture

def bottleneck_lstm(inputs,filters,state):
    input_conv,_ = squeezenext_architecture.squeezenext_unit(inputs, filters*4, 1,True, 1,False)
    concat = tf.concat([input_conv,state[0]],axis=-1)
    gate,_ = squeezenext_architecture.squeezenext_unit(concat, filters, 1,False, 1,False)
    intermediates,_ = squeezenext_architecture.squeezenext_unit(gate, filters*4, 1,True, 1,False)
    in_gate = intermediates[:,:,:,0:filters]*intermediates[:,:,:,filters:filters*2]
    state = (intermediates[:,:,:,filters*2:filters*3]*state[1])+in_gate
    output = intermediates[:,:,:,filters*3:]*state
    return output,state


class ConvolutionalLstm(tf.nn.rnn_cell.RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, model, training,num_classes, scope=None):
        self._model = model
        self._training = training
        self._num_classes = num_classes
        self._scope = tf.name_scope(scope or type(self).__name__)

    @property
    def state_size(self):
        return (tf.constant([7,7,64]), tf.constant([7,7,64]))

    @property
    def output_size(self):
        return self._num_classes



    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(self._scope.name):
            # create model inside the argscope of the model
            with slim.arg_scope(self._model.model_arg_scope(self._training,updates_collections="rnn_batchnorm")):
                predictions, endpoints = self._model(inputs, self._training)
            state = bottleneck_lstm(endpoints["block_3/unit_0"],64,state)
            # output conv and pooling
            net = slim.conv2d(state[0], 128, [1, 1], scope="output_conv")
            endpoints["output_conv"] = net
            net = tf.squeeze(
                slim.avg_pool2d(net, net.get_shape().as_list()[1:3], scope="avg_pool_out", padding="VALID"),
                axis=[1, 2])
            endpoints["avg_pool_out"] = net

            # Fully connected output without biases
            output = slim.fully_connected(net, self._model.num_classes, activation_fn=None, normalizer_fn=None,
                                          biases_initializer=None)
            endpoints["output"] = output
            with tf.control_dependencies(tf.get_collection("rnn_batchnorm")):
                 return predictions, state