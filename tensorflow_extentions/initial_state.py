import tensorflow as tf
slim = tf.contrib.slim
import numpy as np

def make_gaussian_state_initializer(cell,batch_size, deterministic_tensor=None, stddev=0.3):
    learnable_state_shape = cell.state_size
    print(learnable_state_shape)
    init_states = []
    for idx,shape in enumerate(learnable_state_shape):
        print([batch_size]+shape[1].as_list())
        learnable_state_1 = slim.variable('init_{}_state_1'.format(idx),
                                shape=[batch_size]+shape[0].as_list(),
                                initializer=tf.truncated_normal_initializer(stddev=0.09))
        learnable_state_2 = slim.variable('init_{}_state_2'.format(idx),
                                        shape=[batch_size]+shape[1].as_list(),
                                        initializer=tf.truncated_normal_initializer(stddev=0.09))

        if deterministic_tensor is not None:
            init_states.append((learnable_state_1 + tf.random_normal(tf.shape(learnable_state_1),
                                                                    stddev=stddev),
                               learnable_state_2 + tf.random_normal(tf.shape(learnable_state_2),
                                                                    stddev=stddev)))
    return init_states
