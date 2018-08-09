import tensorflow as tf
import tools
from squeezenext_architecture import squeezenext_unit
slim = tf.contrib.slim
from collections import OrderedDict

def class_net(inputs,num_classes,num_anchors):
    height_first_order = False
    input_channels = inputs.get_shape().as_list()[-1]
    net, height_first_order = squeezenext_unit(inputs, input_channels, 1, height_first_order, 1,False)
    net, height_first_order = squeezenext_unit(inputs, input_channels, 1, height_first_order, 1, False)
    net, height_first_order = squeezenext_unit(inputs, input_channels*2, 1, height_first_order, 1, False)
    net, height_first_order = squeezenext_unit(inputs, input_channels*2, 1, height_first_order, 1, False)
    return slim.conv2d(net, num_classes * num_anchors, [1, 1], scope="output_conv",activation_fn=None,normalizer_fn=None)

def box_net(inputs,num_anchors):
    height_first_order = False
    input_channels = inputs.get_shape().as_list()[-1]
    net, height_first_order = squeezenext_unit(inputs, input_channels, 1, height_first_order, 1,False)
    net, height_first_order = squeezenext_unit(inputs, input_channels, 1, height_first_order, 1, False)
    net, height_first_order = squeezenext_unit(inputs, input_channels, 1, height_first_order, 1, False)
    net, height_first_order = squeezenext_unit(inputs, input_channels/2, 1, height_first_order, 1, False)
    return slim.conv2d(net, 4 * num_anchors, [1, 1], scope="output_conv",activation_fn=None,normalizer_fn=None)



class ModelHead(object):
    def __init__(self, params):
        self._num_anchors = params["num_scales"] * len(params['aspect_ratios'])
        self._num_classes = params["num_classes"]

    def model_head(self,endpoints):

        with tf.variable_scope('retinanet'):
            small_input = endpoints["small"][0]
            medium_input = endpoints["medium"][0]
            large_input = endpoints["large"][0]
            fpn = tools.create_feature_pyramid([small_input, medium_input, large_input])
            class_outputs = []
            box_outputs = []
            with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
                for idx,level in enumerate(fpn):
                    class_outputs.append(class_net(level, self._num_classes,self._num_anchors))
            with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
                for idx, level in enumerate(fpn):
                    box_outputs.append(box_net(level,self._num_anchors))
        return class_outputs, box_outputs
