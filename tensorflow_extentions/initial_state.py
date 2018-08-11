import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.python.util import nest

def get_initial_cell_state(cell, initializer, batch_size, dtype):
    """Return state tensor(s), initialized with initializer.
    Args:
      cell: RNNCell.
      batch_size: int, float, or unit Tensor representing the batch size.
      initializer: function with two arguments, shape and dtype, that
          determines how the state is initialized.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` initialized
      according to the initializer.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    with tf.name_scope("state_initializer"):
        state_size = cell.state_size
        init_state,_ = recursive_state_shape_initialization(initializer,state_size,batch_size,dtype,0)
        state_list = []
        for tensor_tuple in init_state:
            state_list.append(tensor_tuple)

    return state_list

def recursive_state_shape_initialization(initializer,state_size,batch_size,dtype,index):
    if nest.is_sequence(state_size[0]):
        results = []
        for size in state_size:
            variable,index = recursive_state_shape_initialization(initializer,size,batch_size,dtype,index)
            results.append(variable)
        if isinstance(state_size, tuple):
            return tuple(results),index
        elif isinstance(state_size, list):
            return list(results),index
        else:
            "error"
    else:
        return initializer(state_size, batch_size, dtype,index),index+1

def make_variable_state_initializer(**kwargs):
    def variable_state_initializer(shape, batch_size, dtype, index):
        args = kwargs.copy()

        if args.get('name'):
            args['name'] = args['name'] + '_' + str(index)
        else:
            args['name'] = 'init_state_' + str(index)

        args['shape'] = shape
        args['dtype'] = dtype
        var = tf.get_variable(**args)
        var = tf.expand_dims(var, 0)
        var = tf.tile(var, tf.stack([batch_size] + [1] * len(shape)))
        var.set_shape([batch_size] + shape)
        return var

    return variable_state_initializer

def make_gaussian_state_initializer(initializer, deterministic_tensor=None, stddev=0.3):
    def gaussian_state_initializer(shape, batch_size, dtype, index):
        init_state = initializer(shape, batch_size, dtype, index)
        if deterministic_tensor is not None:
            return tf.cond(deterministic_tensor,
                lambda: init_state,
                lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev))
        else:
            return init_state + tf.random_normal(tf.shape(init_state), stddev=stddev)
    return gaussian_state_initializer

def zero_state_initializer(shape, batch_size, dtype, index):
    z = tf.zeros(tf.stack([batch_size]+shape), dtype)
    z.set_shape([batch_size]+shape)
    return z