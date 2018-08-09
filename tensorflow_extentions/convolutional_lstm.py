from __future__ import absolute_import


import tensorflow as tf
slim = tf.contrib.slim
import contextlib

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

def bottleneck_lstm(inputs,filters,state):
    input_conv = slim.separable_convolution2d(inputs, None, [3,3],1)
    concat = tf.concat([input_conv,state[0]],axis=-1)
    gate = slim.conv2d(concat,filters,[1,1])
    intermediates = split_separable_conv2d(gate, filters*4)
    in_gate = intermediates[:,:,:,0:filters]*intermediates[:,:,:,filters:filters*2]
    state = (intermediates[:,:,:,filters*2:filters*3]*state[1])+in_gate
    output = intermediates[:,:,:,filters*3:]*state
    return output,state

def calculate_state_sizes(params):
    state_sizes = []
    image_size = params["image_size"]-5
    for level in range(params["min_level"], params["max_level"] + 1):
        state_sizes.append([image_size/2 ** level,image_size/2 ** level])
    return state_sizes
class ConvolutionalLstm(tf.nn.rnn_cell.RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, model, head, training,params, scope=None):
        self._model = model
        self._training = training
        self._model_head = head.model_head
        self._num_classes = params["num_classes"]
        self._scope = tf.name_scope(scope or type(self).__name__)
        self._state_size = calculate_state_sizes(params)
        self._num_filters = params["num_filters"]
        self._num_anchors = params["num_scales"] * len(params['aspect_ratios'])
        self._num_classes = params["num_classes"]

    @property
    def state_size(self):
        state_sizes = [tf.constant(size+[self._num_filters]) for size in self._state_size]
        state_sizes = tuple([(state_size,state_size) for state_size in state_sizes])
        return state_sizes

    @property
    def output_size(self):
        box_sizes = [tf.constant(size + [self._num_anchors * 4]) for size in self._state_size]
        class_sizes = [tf.constant(size + [self._num_anchors*self._num_classes]) for size in self._state_size]
        return (class_sizes,box_sizes)



    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(self._scope.name):
            # create model inside the argscope of the model
            with slim.arg_scope(self._model.model_arg_scope(self._training,updates_collections="rnn_batchnorm")):
                with tf.variable_scope("classifier"):
                    _, classifier_endpoints = self._model(inputs, self._training)
                with tf.variable_scope("bottleneck"):
                    endpoints = {}
                    endpoints["small"] = bottleneck_lstm(classifier_endpoints["block_1/unit_0"],self._num_filters,state[0])
                    endpoints["medium"] = bottleneck_lstm(classifier_endpoints["block_2/unit_0"], self._num_filters, state[1])
                    endpoints["large"] = bottleneck_lstm(classifier_endpoints["block_3/unit_0"], self._num_filters, state[2])
                    # small = slim.conv2d(classifier_endpoints["block_1/unit_0"], self._num_filters, [1, 1])
                    # medium = slim.conv2d(classifier_endpoints["block_2/unit_0"], self._num_filters, [1, 1])
                    # large = slim.conv2d(classifier_endpoints["block_3/unit_0"], self._num_filters, [1, 1])
                    # endpoints["small"] = (small,small)
                    # endpoints["medium"] = (medium,medium)
                    # endpoints["large"] = (large,large)

                state = (  endpoints["small"] ,endpoints["medium"] ,endpoints["large"])

                outputs = self._model_head(endpoints)

            with tf.control_dependencies(tf.get_collection("rnn_batchnorm")):
                 return outputs, state