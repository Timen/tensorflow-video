from __future__ import absolute_import


import tensorflow as tf
slim = tf.contrib.slim
from collections import OrderedDict
class TimeDistributedWrapper(tf.nn.rnn_cell.RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, model,output_shape,endpoints=None, scope=None):
        self._model = model
        self._output_shape = output_shape
        self._scope = tf.name_scope(scope or type(self).__name__)
        self._output_endpoints = endpoints

    @property
    def state_size(self):
        return []

    @property
    def output_size(self):
        if isinstance(self._output_shape[0][0],list):
            output_shapes = []
            for shapes in self._output_shape:
                output_shapes.append([tf.constant(shape) for shape in shapes])
            return output_shapes
        else:
            return [tf.constant(shape) for shape in self._output_shape]



    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        scope_name = scope if scope else self._scope.name
        with tf.variable_scope(scope_name):
            self.scope_string = tf.get_variable_scope().name
            # create model inside the argscope of the model
            with slim.arg_scope([slim.batch_norm], **{"updates_collections": scope_name + "/rnn_batchnorm"}):
                if self._output_endpoints:
                    _,endpoints = self._model(inputs)
                    predictions = []
                    for key in self._output_endpoints:
                        predictions.append(endpoints[key])
                else:
                    predictions = self._model(inputs)

            with tf.control_dependencies(tf.get_collection("rnn_batchnorm")):
                 return predictions, []



def time_distributed( inputs, sequence_length, model, output_shape, endpoints=None, scope=None, dtype=tf.float32,return_scope_string=False):
    time_cell = TimeDistributedWrapper(model, output_shape, endpoints=endpoints, scope=scope)
    predictions, _ = tf.nn.dynamic_rnn(
        cell=time_cell,
        dtype=dtype,
        initial_state=[],
        sequence_length=sequence_length,
        inputs=inputs)
    if endpoints:
        prediction_endpoints = OrderedDict()
        for key,prediction in zip(endpoints,predictions):
            prediction_endpoints[key] = prediction
        predictions = prediction_endpoints
    if return_scope_string:
        return predictions,time_cell.scope_string
    else:
        return predictions