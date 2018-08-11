from __future__ import absolute_import

import tensorflow as tf
import os
from collections import OrderedDict
import anchors
slim = tf.contrib.slim
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

def define_first_dim(tensor_dict,dim_size):
    """
    Define the first dim keeping the remaining dims the same
    :param tensor_dict:
        Dictionary of tensors
    :param dim_size:
        Size of first dimention
    :return:
        Dictionary of dimensions with the first dim defined as dim_size
    """
    for key, value in tensor_dict.iteritems():
        if isinstance(value, dict):
            nested_shape_dict = OrderedDict()
            for nested_key, tensor in value.iteritems():
                shape = tensor.get_shape().as_list()[1:]
                nested_shape_dict[nested_key] = tf.reshape(tensor, [dim_size] + shape)
            tensor_dict[key] = nested_shape_dict
        else:
            shape = value.get_shape().as_list()[1:]
            tensor_dict[key] = tf.reshape(value, [dim_size] + shape)
    return tensor_dict


def combine_dims(tensor,dims):
    tensor_shape = tensor.get_shape().as_list()
    sizes = [tensor_shape[i] for i in dims]
    combined_dim = reduce(lambda x,y:x*y,sizes)
    new_shape = []
    add_combined = False
    for idx,dim in enumerate(tensor_shape):
        if not idx in dims:
            new_shape.append(dim)
        elif not add_combined:
            new_shape.append(combined_dim)
            add_combined = True
    return tf.reshape(tensor,new_shape)


def get_pred_results(cls_outputs_dict,box_outputs_dict, params):
    input_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                    params['num_scales'],
                                    params['aspect_ratios'],
                                    params['anchor_scale'],
                                    (params['image_size'] - 5))
    anchor_labeler = anchors.AnchorLabeler(input_anchors, params['num_classes'])

    return tf.map_fn(anchor_labeler.generate_detections,(cls_outputs_dict,box_outputs_dict),dtype=tf.float32)

def yxyx_to_xywh(bboxes):
    ymin = bboxes[:,:,0:1]
    xmin = bboxes[:, :, 1:2]
    ymax = bboxes[:, :, 2:3]
    xmax = bboxes[:, :, 3:4]
    width = xmax-xmin
    height = ymax-ymin

    return tf.concat([xmin,ymin,width,height],axis=-1)



def draw_box_predictions(images,predictions,labels,params,sequence_length,num_display=4,max_images=32):
    if (num_display*sequence_length) > max_images:
        num_display = max_images/sequence_length

    for i in range(0,num_display):
        cls_outputs_dict = {}
        box_outputs_dict = {}

        # image = tf.image.draw_bounding_boxes(images[i, :, :, :, :],labels["boxes"][i, :, :,:]/params['image_size'])
        image = images[i, :, :, :, :]
        for idx,(cls_outputs,box_outputs) in enumerate(predictions):
            cls_outputs_dict[params["min_level"]+idx] = cls_outputs[i,:,:,:,:]
            box_outputs_dict[params["min_level"]+idx] = box_outputs[i,:,:,:,:]
        boxes = get_pred_results(cls_outputs_dict,box_outputs_dict,params)[:,:,0:4]
        tf.summary.image("sequence_{}".format(i), tf.image.draw_bounding_boxes(image,boxes/params['image_size']), max_outputs=sequence_length)

def eval_predictions(predictions,params):

    cls_outputs_dict = {}
    box_outputs_dict = {}

    for idx,(cls_outputs,box_outputs) in enumerate(zip(*predictions)):
        cls_outputs_dict[params["min_level"]+idx] = combine_dims(cls_outputs,[0,1])
        box_outputs_dict[params["min_level"]+idx] = combine_dims(box_outputs, [0, 1])
    results = get_pred_results(cls_outputs_dict,box_outputs_dict,params)
    results = tf.concat([tf.zeros_like(results[:,:,0:1]),yxyx_to_xywh(results[:,:,0:4]),results[:,:,4:]],axis=-1)
    return results

def eval_labels(labels):
    classes = combine_dims(labels["classes"],[0,1])
    boxes = combine_dims(labels["boxes"], [0, 1])
    return tf.concat([boxes,classes],axis=-1)



def create_feature_pyramid(inputs, default_length=128):
    inputs = [ slim.conv2d(tensor,default_length,[1,1]) for tensor in inputs]
    with tf.name_scope("feature_pyramid"):
        for idx, input_image in enumerate(inputs):
            if idx+1 < len(inputs):
                next_scale =inputs[idx+1]
                size = next_scale.get_shape().as_list()[1:3]
                upsampled = tf.image.resize_nearest_neighbor(input_image,size)
                inputs[idx+1] =  upsampled+next_scale
        return inputs


def get_checkpoint_step(checkpoint_dir):
    """
    Get step at which checkpoint was saved from file name
    :param checkpoint_dir:
        Directory containing a checkpoint
    :return:
        Step at which checkpoint was saved
    """
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt is None:
        return None
    else:
        return int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])


def get_or_create_global_step():
    """
    Checks if global step variable exists otherwise creates it
    :return:
    Global step tensor
    """
    global_step = tf.train.get_global_step()
    if global_step is None:
        global_step = tf.train.create_global_step()
    return global_step

def warmup_phase(learning_rate_schedule,base_lr,warmup_steps,warmup_learning_rate):
    """
    Ramps up the learning rate from warmup_learning_rate till base_lr in warmup_steps before
    switching to learning_rate_schedule.
    The warmup is linear and calculated using the below functions.
    slope = (base_lr - warmup_learning_rate) / warmup_steps
    warmup_rate = slope * global_step + warmup_learning_rate

    :param learning_rate_schedule:
        A regular learning rate schedule such as stepwise,exponential decay etc
    :param base_lr:
        The learning rate to which to ramp up to
    :param warmup_steps:
        The number of steps of the warmup phase
    :param warmup_learning_rate:
        The learning rate from which to start ramping up to base_lr
    :return:
        Warmup learning rate for global step <  warmup_steps else returns learning_rate_schedule
    """
    with tf.name_scope("warmup_learning_rate"):
        global_step = tf.cast(get_or_create_global_step(),tf.float32)
        if warmup_steps > 0:
            if base_lr < warmup_learning_rate:
                raise ValueError('learning_rate_base must be larger or equal to '
                                 'warmup_learning_rate.')
            slope = (learning_rate_schedule - warmup_learning_rate) / warmup_steps
            warmup_rate = slope * global_step + warmup_learning_rate
            learning_rate_schedule = tf.where(global_step < warmup_steps, warmup_rate,
                                     learning_rate_schedule)
        return learning_rate_schedule
