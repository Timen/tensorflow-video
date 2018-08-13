# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""RetinaNet anchor definition.

This module implements RetinaNet anchor described in:

T.-Y. Lin, P. Goyal, R. Girshick, K. He,  and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np
import tensorflow as tf
from object_detection import argmax_matcher
from object_detection import box_list
from object_detection import faster_rcnn_box_coder
from object_detection import region_similarity_calculator
from object_detection import target_assigner

# The minimum score to consider a logit for identifying detections.
MIN_CLASS_SCORE = -0.0

# The score for a dummy detection
_DUMMY_DETECTION_SCORE = -1e5

# The maximum number of (anchor,class) pairs to keep for non-max suppression.
MAX_DETECTION_POINTS = 200

# The maximum number of detections per image.
MAX_DETECTIONS_PER_IMAGE = 20


def sigmoid(x):
    """Sigmoid function for use with Numpy for CPU evaluation."""
    return 1 / (1 + np.exp(-x))


def safe_exp(w, thresh):
  """Safe exponential function for tensors."""

  slope = np.exp(thresh)
  with tf.variable_scope('safe_exponential'):
    lin_bool = w > thresh
    lin_region = tf.to_float(lin_bool)

    lin_out = slope*(w - thresh + 1.)
    exp_out = tf.exp(tf.where(lin_bool, tf.zeros_like(w), w))

    out = lin_region*lin_out + (1.-lin_region)*exp_out
  return out

def decode_box_outputs(rel_codes, anchors,exp_thresh=5):
    """Transforms relative regression coordinates to absolute positions.

    Network predictions are normalized and relative to a given anchor; this
    reverses the transformation and outputs absolute coordinates for the input
    image.

    Args:
      rel_codes: box regression targets.
      anchors: anchors on all feature levels.
    Returns:
      outputs: bounding boxes.

    """

    ymin,xmin,ymax,xmax = tf.unstack(anchors,axis=1)
    wa = xmax - xmin
    ha = ymax - ymin
    ycenter_a = ymin + ha / 2.
    xcenter_a = xmin + wa / 2.

    ty, tx, th, tw = tf.unstack(rel_codes,axis=1)


    w = safe_exp(tw,exp_thresh) * wa
    h = safe_exp(th,exp_thresh) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return tf.stack([ymin, xmin, ymax, xmax],axis=1)


def _generate_anchor_configs(min_level, max_level, num_scales, aspect_ratios):
    """Generates mapping from output level to a list of anchor configurations.

    A configuration is a tuple of (num_anchors, scale, aspect_ratio).

    Args:
        min_level: integer number of minimum level of the output feature pyramid.
        max_level: integer number of maximum level of the output feature pyramid.
        num_scales: integer number representing intermediate scales added
          on each level. For instances, num_scales=2 adds two additional
          anchor scales [2^0, 2^0.5] on each level.
        aspect_ratios: list of tuples representing the aspect raito anchors added
          on each level. For instances, aspect_ratios =
          [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.
    Returns:
      anchor_configs: a dictionary with keys as the levels of anchors and
        values as a list of anchor configuration.
    """
    anchor_configs = {}
    for level in range(min_level, max_level + 1):
        anchor_configs[level] = []
        for scale_octave in range(num_scales):
            for aspect in aspect_ratios:
                anchor_configs[level].append(
                    (2 ** level, scale_octave / float(num_scales), aspect))
    return anchor_configs


def _generate_anchor_boxes(image_size, anchor_scale, anchor_configs):
    """Generates multiscale anchor boxes.

    Args:
      image_size: integer number of input image size. The input image has the
        same dimension for width and height. The image_size should be divided by
        the largest feature stride 2^max_level.
      anchor_scale: float number representing the scale of size of the base
        anchor to the feature stride 2^level.
      anchor_configs: a dictionary with keys as the levels of anchors and
        values as a list of anchor configuration.
    Returns:
      anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
        feature levels.
    Raises:
      ValueError: input size must be the multiple of largest feature stride.
    """
    boxes_all = []
    for _, configs in anchor_configs.items():
        boxes_level = []
        for config in configs:
            stride, octave_scale, aspect = config
            if image_size % stride != 0:
                raise ValueError("input size must be divided by the stride.")
            base_anchor_size = anchor_scale * stride * 2 ** octave_scale
            anchor_size_x_2 = base_anchor_size * aspect[0] / 2.0
            anchor_size_y_2 = base_anchor_size * aspect[1] / 2.0

            x = np.arange(stride / 2, image_size, stride)
            y = np.arange(stride / 2, image_size, stride)
            xv, yv = np.meshgrid(x, y)
            xv = xv.reshape(-1)
            yv = yv.reshape(-1)

            boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                               yv + anchor_size_y_2, xv + anchor_size_x_2))
            boxes = np.swapaxes(boxes, 0, 1)
            boxes_level.append(np.expand_dims(boxes, axis=1))
        # concat anchors on the same level to the reshape NxAx4
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))

    anchor_boxes = np.vstack(boxes_all)
    return anchor_boxes


def _generate_detections(cls_outputs, box_outputs, anchor_boxes):
    """Generates detections with RetinaNet model outputs and anchors.

    Args:
      cls_outputs: a numpy array with shape [N, num_classes], which stacks class
        logit outputs on all feature levels. The N is the number of total anchors
        on all levels. The num_classes is the number of classes predicted by the
        model.
      box_outputs: a numpy array with shape [N, 4], which stacks box regression
        outputs on all feature levels. The N is the number of total anchors on all
        levels.
      anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
        feature levels. The N is the number of total anchors on all levels.
      image_id: an integer number to specify the image id.
      image_scale: a float tensor representing the scale between original image
        and input image for the detector. It is used to rescale detections for
        evaluating with the original groundtruth annotations.
    Returns:
      detections: detection results in a tensor with each row representing
        [image_id, x, y, width, height, score, class]
    """
    confidences = tf.reduce_max(cls_outputs,axis=1)
    mask = tf.greater(confidences , MIN_CLASS_SCORE)
    anchor_boxes = tf.boolean_mask(anchor_boxes, mask)
    class_idx = tf.reshape(tf.cast(tf.boolean_mask(tf.argmax(box_outputs,axis=-1), mask),tf.float32),[-1,1])
    confidences =  tf.reshape(tf.boolean_mask(confidences, mask),[-1,1])
    box_outputs = tf.boolean_mask(box_outputs, mask)

    num_elements = tf.shape(box_outputs)[0]
    box_outputs = tf.cond(num_elements > 0, lambda:box_outputs,lambda:tf.zeros([MAX_DETECTION_POINTS,4],dtype=tf.float32))
    confidences = tf.cond(num_elements > 0, lambda:confidences,lambda:tf.zeros([MAX_DETECTION_POINTS,1],dtype=tf.float32))
    class_idx = tf.cond(num_elements > 0, lambda:class_idx, lambda:tf.zeros([MAX_DETECTION_POINTS,1],dtype=tf.float32))
    anchor_boxes = tf.cond(num_elements > 0, lambda:anchor_boxes, lambda:tf.zeros([MAX_DETECTION_POINTS,4],dtype=tf.float32))

    paddings = tf.cast(((0, tf.clip_by_value(MAX_DETECTION_POINTS - num_elements,0,MAX_DETECTION_POINTS)), (0, 0)), tf.int32)
    confidences = tf.sigmoid(confidences)
    confidences = tf.pad(confidences, paddings, constant_values=0.0)
    class_idx = tf.pad(class_idx, paddings, constant_values=-1.0)
    box_outputs = tf.pad(box_outputs, paddings, constant_values=0.0)
    anchor_boxes = tf.pad(anchor_boxes, paddings, constant_values=0.0)
    confidences, indices = tf.nn.top_k(confidences[:,0],k=MAX_DETECTION_POINTS)

    anchor_boxes = tf.reshape(tf.gather(anchor_boxes, indices),[MAX_DETECTION_POINTS,4])
    class_idx = tf.gather(class_idx, indices)
    box_outputs =  tf.reshape(tf.gather(box_outputs, indices),[MAX_DETECTION_POINTS,4])

    # apply bounding box regression to anchors
    boxes = decode_box_outputs(box_outputs,anchor_boxes)
    nms_indices = tf.image.non_max_suppression(boxes,confidences,MAX_DETECTIONS_PER_IMAGE)

    boxes = tf.gather(boxes,nms_indices)
    class_idx = tf.reshape(tf.gather(class_idx, nms_indices),[-1,1])
    confidences = tf.reshape(tf.gather(confidences, nms_indices),[-1,1])

    num_boxes = tf.shape(boxes)[0]
    paddings = tf.cast(((0,MAX_DETECTIONS_PER_IMAGE - num_boxes),(0,0)),tf.int32)

    boxes = tf.pad(boxes,paddings,constant_values=0.0)
    confidences = tf.pad(confidences,paddings,constant_values=0.0)
    class_idx = tf.pad(tf.cast(class_idx,tf.float32), paddings,constant_values=-1.0)
    return tf.concat([boxes,confidences,class_idx],axis=-1)


class Anchors(object):
    """RetinaNet Anchors class."""

    def __init__(self, min_level, max_level, num_scales, aspect_ratios,
                 anchor_scale, image_size):
        """Constructs multiscale RetinaNet anchors.

        Args:
          min_level: integer number of minimum level of the output feature pyramid.
          max_level: integer number of maximum level of the output feature pyramid.
          num_scales: integer number representing intermediate scales added
            on each level. For instances, num_scales=2 adds two additional
            anchor scales [2^0, 2^0.5] on each level.
          aspect_ratios: list of tuples representing the aspect raito anchors added
            on each level. For instances, aspect_ratios =
            [(1, 1), (1.4, 0.7), (0.7, 1.4)] adds three anchors on each level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
        """
        self.min_level = min_level
        self.max_level = max_level
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = anchor_scale
        self.image_size = image_size
        self.config = self._generate_configs()
        self.boxes = self._generate_boxes()

    def _generate_configs(self):
        """Generate configurations of anchor boxes."""
        return _generate_anchor_configs(self.min_level, self.max_level,
                                        self.num_scales, self.aspect_ratios)

    def _generate_boxes(self):
        """Generates multiscale anchor boxes."""
        boxes = _generate_anchor_boxes(self.image_size, self.anchor_scale,
                                       self.config)
        boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
        return boxes

    def get_anchors_per_location(self):
        return self.num_scales * len(self.aspect_ratios)


class AnchorLabeler(object):
    """Labeler for multiscale anchor boxes."""

    def __init__(self, anchors, num_classes, match_threshold=0.5):
        """Constructs anchor labeler to assign labels to anchors.

        Args:
          anchors: an instance of class Anchors.
          num_classes: integer number representing number of classes in the dataset.
          match_threshold: float number between 0 and 1 representing the threshold
            to assign positive labels for anchors.
        """
        similarity_calc = region_similarity_calculator.IouSimilarity()
        matcher = argmax_matcher.ArgMaxMatcher(
            match_threshold,
            unmatched_threshold=match_threshold,
            negatives_lower_than_unmatched=True,
            force_match_for_each_row=True)
        box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder()

        self._target_assigner = target_assigner.TargetAssigner(
            similarity_calc, matcher, box_coder)
        self._anchors = anchors
        self._match_threshold = match_threshold
        self._num_classes = num_classes

    def _unpack_labels(self, labels):
        """Unpacks an array of labels into multiscales labels."""
        labels_unpacked = OrderedDict()
        anchors = self._anchors
        count = 0
        for level in range(anchors.min_level, anchors.max_level + 1):
            feat_size = int(anchors.image_size / 2 ** level)
            steps = feat_size ** 2 * anchors.get_anchors_per_location()
            indices = tf.range(count, count + steps)
            count += steps
            labels_unpacked[level] = tf.reshape(
                tf.gather(labels, indices), [feat_size, feat_size, -1])
        return labels_unpacked

    def label_anchors(self, gt_boxes, gt_labels):
        """Labels anchors with ground truth inputs.

        Args:
          gt_boxes: A float tensor with shape [N, 4] representing groundtruth boxes.
            For each row, it stores [y0, x0, y1, x1] for four corners of a box.
          gt_labels: A integer tensor with shape [N, 1] representing groundtruth
            classes.
        Returns:
          cls_targets_dict: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, num_anchors]. The height_l and width_l
            represent the dimension of class logits at l-th level.
          box_targets_dict: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, num_anchors * 4]. The height_l and
            width_l represent the dimension of bounding box regression output at
            l-th level.
          num_positives: scalar tensor storing number of positives in an image.
        """
        gt_box_list = box_list.BoxList(gt_boxes)
        anchor_box_list = box_list.BoxList(self._anchors.boxes)

        # cls_weights, box_weights are not used
        cls_targets, _, box_targets, _, matches = self._target_assigner.assign(
            anchor_box_list, gt_box_list, gt_labels)

        # class labels start from 1 and the background class = -1
        cls_targets -= 1
        cls_targets = tf.cast(cls_targets, tf.int32)

        # Unpack labels.
        cls_targets_dict = self._unpack_labels(cls_targets)
        box_targets_dict = self._unpack_labels(box_targets)
        num_positives = tf.reduce_sum(
            tf.cast(tf.not_equal(matches.match_results, -1), tf.float32))

        return cls_targets_dict, box_targets_dict, num_positives

    def generate_detections(self, elems):
        cls_outputs, box_outputs = tuple(elems)
        cls_outputs_all = []
        box_outputs_all = []
        for level in range(self._anchors.min_level, self._anchors.max_level + 1):
            cls_outputs_all.append(
                tf.reshape(cls_outputs[level], [-1, self._num_classes]))
            box_outputs_all.append(tf.reshape(box_outputs[level], [-1, 4]))
        cls_outputs_all = tf.concat(cls_outputs_all, 0)
        box_outputs_all = tf.concat(box_outputs_all, 0)
        return _generate_detections(cls_outputs_all, box_outputs_all, self._anchors.boxes)

