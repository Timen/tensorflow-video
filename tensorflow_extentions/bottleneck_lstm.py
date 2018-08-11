from __future__ import absolute_import


import tensorflow as tf
slim = tf.contrib.slim
import contextlib
import collections
import pprint
import tensorflow_extentions as tfe
@contextlib.contextmanager
def _v1_compatible_scope_naming(scope):
  if scope is None:  # Create uniqified separable blocks.
    with tf.variable_scope(None, default_name='separable') as s, \
         tf.name_scope(s.original_name_scope):
      yield ''
  else:
    # We use scope_depthwise, scope_pointwise for compatibility with V1 ckpts.
    # which provide numbered scopes.
    scope += '_'
    yield scope

def mobilenet_v1_arg_scope(
    is_training=True,
    weight_decay=0.00004,
    stddev=0.09,
    regularize_depthwise=False,
    batch_norm_decay=0.9997,
    batch_norm_epsilon=0.001,
    batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
  """Defines the default MobilenetV1 arg scope.
  Args:
    is_training: Whether or not we're training the model. If this is set to
      None, the parameter is not added to the batch_norm arg_scope.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    regularize_depthwise: Whether or not apply regularization on depthwise.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
  Returns:
    An `arg_scope` to use for the mobilenet v1 model.
  """
  batch_norm_params = {
      'center': True,
      'scale': True,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'updates_collections': batch_norm_updates_collections,
  }
  if is_training is not None:
    batch_norm_params['is_training'] = is_training

  # Set weight_decay for weights in Conv and DepthSepConv layers.
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  if regularize_depthwise:
    depthwise_regularizer = regularizer
  else:
    depthwise_regularizer = None
  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_init,
                      activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
        with slim.arg_scope([slim.separable_conv2d],
                            weights_regularizer=depthwise_regularizer) as sc:
          return sc

@slim.add_arg_scope
def split_separable_conv2d(input_tensor,
                           num_outputs,
                           scope=None,
                           normalizer_fn=None,
                           stride=1,
                           rate=1,
                           endpoints=None):
  """Separable mobilenet V1 style convolution.
  Depthwise convolution, with default non-linearity,
  followed by 1x1 depthwise convolution.  This is similar to
  slim.separable_conv2d, but differs in tha it applies batch
  normalization and non-linearity to depthwise. This  matches
  the basic building of Mobilenet Paper
  (https://arxiv.org/abs/1704.04861)
  Args:
    input_tensor: input
    num_outputs: number of outputs
    scope: optional name of the scope. Note if provided it will use
    scope_depthwise for deptwhise, and scope_pointwise for pointwise.
    normalizer_fn: which normalizer function to use for depthwise/pointwise
    stride: stride
    rate: output rate (also known as dilation rate)
    endpoints: optional, if provided, will export additional tensors to it.
    use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
      inputs so that the output dimensions are the same as if 'SAME' padding
      were used.
  Returns:
    output tesnor
  """

  with _v1_compatible_scope_naming(scope) as scope:
    dw_scope = scope + 'depthwise'
    endpoints = endpoints if endpoints is not None else {}
    kernel_size = [3, 3]
    padding = 'SAME'
    net = slim.separable_conv2d(
        input_tensor,
        None,
        kernel_size,
        depth_multiplier=1,
        stride=stride,
        rate=rate,
        normalizer_fn=normalizer_fn,
        padding=padding,
        scope=dw_scope)

    endpoints[dw_scope] = net

    pw_scope = scope + 'pointwise'
    net = slim.conv2d(
        net,
        num_outputs, [1, 1],
        stride=1,
        normalizer_fn=normalizer_fn,
        scope=pw_scope)
    endpoints[pw_scope] = net
  return net

def bottleneck_lstm(inputs,state,filters):
    """
    Bottleneck lstm from:
    https://arxiv.org/pdf/1711.06368.pdf

    :param inputs:
        Input featuremap [B,H,W,C]
    :param state:
        Previous state [B,H,W,C]
    :param filters:
        number of output channels
    :return:
        tuple containing output and state featuremaps
    """
    with tf.name_scope("BottleneckLSTM"):
        input_conv = slim.separable_convolution2d(inputs, None, [3,3],1)
        concat = tf.concat([input_conv,state[0]],axis=-1)
        gate = slim.conv2d(concat,filters,[1,1])
        intermediates = split_separable_conv2d(gate, filters*4)
        in_gate = intermediates[:,:,:,:filters]*intermediates[:,:,:,filters:filters*2]
        state = (intermediates[:,:,:,filters*2:filters*3]*state[1])+in_gate
        output = intermediates[:,:,:,filters*3:]*state
    return output,state


class ConvolutionalLstm(tf.nn.rnn_cell.RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, input_shapes,num_filters=None, scope=None):
        super(ConvolutionalLstm, self).__init__()
        self._endpoints = []
        if isinstance(input_shapes, collections.OrderedDict):
            self._input_shapes = []
            for key,tensor in input_shapes.iteritems():
                self._input_shapes.append(tensor.get_shape().as_list()[-3:])
                self._endpoints.append(key)
        elif isinstance(input_shapes, list):
            self._input_shapes = input_shapes
        else:
            raise TypeError("input_shapes must be dict of tensors or list of shapes")
        self._scope = tf.name_scope(scope or type(self).__name__)
        self.output_filters = []
        self._output_shapes = []
        if num_filters is None:
            for shape in  self._input_shapes:
                output_channels = shape[-1]/4
                self.output_filters.append(output_channels)
                self._output_shapes.append(shape[:-1]+[output_channels])
        else:
            if isinstance(num_filters, list):
                assert len(num_filters) == len( self._input_shapes), "num_filters must be specified for each input"
                for output_channels,shape in zip(input_shapes,num_filters):
                    self.output_filters.append(output_channels)
                    self._output_shapes.append(shape[:-1] + [output_channels])
            else:
                for shape in  self._input_shapes:
                    self.output_filters.append(num_filters)
                    self._output_shapes.append(shape[:-1]+[num_filters])


    @property
    def state_size(self):
        states = [(shape,shape)for shape in self._output_shapes]
        return states

    @property
    def output_size(self):
        outputs = [tf.constant(shape) for shape in self._output_shapes]
        return outputs


    def __call__(self, inputs, states, scope=None):
        """Long short-term memory cell (LSTM)."""
        scope_name = scope if scope else self._scope.name
        with tf.variable_scope(scope_name):

            if len(self._endpoints) > 0:
                inputs_list = []
                for endpoint in self._endpoints:
                    inputs_list.append(inputs[endpoint])
                inputs = inputs_list

            with slim.arg_scope(mobilenet_v1_arg_scope(batch_norm_updates_collections=scope_name+"/rnn_batchnorm")):
                if isinstance(inputs, collections.Iterable):
                    outputs = []
                    output_states = []
                    for input_tensor,num_filters,state in zip(inputs,self.output_filters,states):
                        result = bottleneck_lstm(input_tensor,state,num_filters)
                        outputs.append(result[0])
                        output_states.append(result)
            with tf.control_dependencies(tf.get_collection("rnn_batchnorm")):
                return outputs, output_states

    def zero_state(self, batch_size, dtype):
        initializer = tfe.initial_state.zero_state_initializer
        return tfe.initial_state.get_initial_cell_state(self, initializer, batch_size, tf.float32)