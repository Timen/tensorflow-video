import tensorflow as tf
import tools
from collections import OrderedDict


def focal_loss(logits, targets, alpha, gamma, normalizer,mask_flat):
    """Compute the focal loss between `logits` and the golden `target` values.
    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    Args:
      logits: A float32 tensor of size
        [batch, height_in, width_in, num_predictions].
      targets: A float32 tensor of size
        [batch, height_in, width_in, num_predictions].
      alpha: A float32 scalar multiplying alpha to the loss from positive examples
        and (1-alpha) to the loss from negative examples.
      gamma: A float32 scalar modulating loss from hard and easy examples.
      normalizer: A float32 scalar normalizes the total loss from all examples.
    Returns:
      loss: A float32 scalar representing normalized total loss.
    """
    with tf.name_scope('focal_loss'):
        positive_label_mask = tf.equal(targets, 1.0)
        cross_entropy = (
            tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
        probs = tf.sigmoid(logits)
        probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
        # With small gamma, the implementation could produce NaN during back prop.
        modulator = tf.pow(1.0 - probs_gt, gamma)
        loss = modulator * cross_entropy
        weighted_loss = tf.where(positive_label_mask, alpha * loss,
                                 (1.0 - alpha) * loss)*mask_flat
        total_loss = tf.reduce_sum(weighted_loss)
        total_loss /= normalizer
    return total_loss


def _classification_loss(cls_outputs,
                         cls_targets,
                         num_positives,
                         mask_flat,
                         alpha=0.25,
                         gamma=2.0):
    """Computes classification loss."""
    normalizer = num_positives
    classification_loss = focal_loss(cls_outputs, cls_targets, alpha, gamma,
                                     normalizer,mask_flat)
    return classification_loss


def _box_loss(box_outputs, box_targets, num_positives,mask_flat, delta=0.1):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
    normalizer = num_positives * 4.0
    mask = tf.cast(tf.not_equal(box_targets, 0.0),tf.float32)*mask_flat
    box_loss = tf.losses.huber_loss(
        box_targets,
        box_outputs,
        weights=mask,
        delta=delta,
        reduction=tf.losses.Reduction.SUM)
    box_loss /= normalizer
    return box_loss


def structure_for_loss(predictions, params):
    class_dict = OrderedDict()
    box_dict = OrderedDict()
    min_level = params["min_level"]
    for idx, output in enumerate(predictions):
        class_dict[min_level + idx] = tools.combine_dims(output[0], [0, 1])
        box_dict[min_level + idx] = tools.combine_dims(output[1], [0, 1])
    return class_dict,box_dict

def detection_loss(predictions, labels, params):
    """Computes total detection loss.
    Computes total detection loss including box and class loss from all levels.
    Args:
      cls_outputs: an OrderDict with keys representing levels and values
        representing logits in [batch_size, height, width, num_anchors].
      box_outputs: an OrderDict with keys representing levels and values
        representing box regression targets in
        [batch_size, height, width, num_anchors * 4].
      labels: the dictionary that returned from dataloader that includes
        groundturth targets.
      params: the dictionary including training parameters specified in
        default_haprams function in this file.
    Returns:
      total_loss: an integar tensor representing total loss reducing from
        class and box losses from all levels.
      cls_loss: an integar tensor representing total class loss.
      box_loss: an integar tensor representing total box regression loss.
    """
    with tf.name_scope("detection_loss"):
        # Sum all positives in a batch for normalization and avoid zero
        # num_positives_sum, which would lead to inf loss during training
        input_shape = predictions[0][0].get_shape().as_list()
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        num_positives_batch = tf.reduce_mean(labels["num_positives"])
        mean_positive = tf.reshape(
            tf.tile(tf.expand_dims(num_positives_batch, 0), [
                batch_size*sequence_length,
            ]), [batch_size*sequence_length, 1])
        num_positives_sum = tf.reduce_sum(mean_positive) + 1.0
        cls_outputs, box_outputs = structure_for_loss(predictions, params)
        levels = cls_outputs.keys()

        cls_losses = []
        box_losses = []
        mask_flat = tf.reshape(labels["loss_masks"], [-1,1,1,1])
        for level in levels:
            # Onehot encoding for classification labels.
            cls_targets_at_level = tf.one_hot(
                tools.combine_dims(labels['cls_targets'][level],[0,1]),
                params['num_classes'])
            bs, width, height, _, _ = cls_targets_at_level.get_shape().as_list()
            cls_targets_at_level = tf.reshape(cls_targets_at_level,
                                              [bs, width, height, -1])
            box_targets_at_level = tools.combine_dims(labels['box_targets'][level],[0,1])
            cls_losses.append(
                _classification_loss(
                    cls_outputs[level],
                    cls_targets_at_level,
                    num_positives_sum,
                    mask_flat,
                    alpha=params['alpha'],
                    gamma=params['gamma']))
            box_losses.append(
                _box_loss(
                    box_outputs[level],
                    box_targets_at_level,
                    num_positives_sum,
                    mask_flat,
                    delta=params['delta']))

        cls_loss = tf.add_n(cls_losses)
        box_loss = tf.add_n(box_losses)
        total_loss = cls_loss + params['box_loss_weight'] * box_loss
    return total_loss, cls_loss, box_loss
