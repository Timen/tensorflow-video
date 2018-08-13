import tensorflow as tf
import tools
from squeezenext_architecture import squeezenext_unit,arg_scope
slim = tf.contrib.slim

def class_net(inputs,num_classes,num_anchors,num_unit_channels=None):
    with tf.variable_scope('class_head'):
        height_first_order = False
        if not num_unit_channels:
            input_channels = inputs.get_shape().as_list()[-1]
        else:
            input_channels = num_unit_channels
        net, height_first_order = squeezenext_unit(inputs, input_channels, 1, height_first_order, 1,False)
        net, height_first_order = squeezenext_unit(net, input_channels, 1, height_first_order, 1, False)
        net, height_first_order = squeezenext_unit(net, input_channels*2, 1, height_first_order, 1, False)
        net, height_first_order = squeezenext_unit(net, input_channels*2, 1, height_first_order, 1, False)
        return slim.conv2d(net, num_classes * num_anchors, [1, 1], scope="output_conv",activation_fn=None,normalizer_fn=None)

def box_net(inputs,num_anchors):
    with tf.variable_scope('box_head'):
        height_first_order = False
        input_channels = inputs.get_shape().as_list()[-1]
        net, height_first_order = squeezenext_unit(inputs, input_channels, 1, height_first_order, 1,False)
        net, height_first_order = squeezenext_unit(net, input_channels, 1, height_first_order, 1, False)
        net, height_first_order = squeezenext_unit(net, input_channels, 1, height_first_order, 1, False)
        net, height_first_order = squeezenext_unit(net, input_channels/2, 1, height_first_order, 1, False)
        return slim.conv2d(net, 4 * num_anchors, [1, 1], scope="output_conv",activation_fn=None,normalizer_fn=None)



class ModelHead(object):
    def __init__(self, params):
        self._num_anchors = params["num_scales"] * len(params['aspect_ratios'])
        self._num_classes = params["num_classes"]

    def __call__(self,inputs):

        with tf.variable_scope('retinanet'):
            small_input = inputs[0]
            medium_input = inputs[1]
            large_input = inputs[2]
            fpn = tools.create_feature_pyramid([small_input, medium_input, large_input])

            outputs = []
            for idx,level in enumerate(fpn):
                with tf.variable_scope('level_{}'.format(idx)):
                    outputs.append((class_net(level, self._num_classes,self._num_anchors),box_net(level, self._num_anchors)))
        return outputs

    def model_arg_scope(self,is_training):
        return arg_scope(is_training)

    def output_size(self,inputs):
        output_shapes = []
        for tensor in inputs:
            inputs_size = tensor.get_shape().as_list()[-3:-1]
            output_shapes.append((inputs_size+[self._num_classes * self._num_anchors ],inputs_size+[4 * self._num_anchors ]))
        return output_shapes