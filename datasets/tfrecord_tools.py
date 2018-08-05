import tensorflow as tf
import six
import numpy as np

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if six.PY3 and isinstance(value, six.text_type):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def add_float_list_feature(feature,list_val):
    if not isinstance(list_val, list):
        list_val = [list_val]
    for value in np.array(list_val).flatten():
        feature.float_list.value.append(value)


def add_int64_list_feature(feature,list_val):
    if not isinstance(list_val, list):
        list_val = [list_val]
    for value in np.array(list_val).flatten():
        feature.int64_list.value.append(value)

def convert_to_example(filename, sequence_features, context_features):
    """Build an Example proto for an example.

    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
      human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
      bbox: list of bounding boxes; each box is a list of integers
        specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong to
        the same label as the image label.
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """

    example = tf.train.SequenceExample()

    # setup context
    example.context.feature["length"].int64_list.value.append(context_features["length"])
    example.context.feature["width"].int64_list.value.append(context_features["width"])
    example.context.feature["height"].int64_list.value.append(context_features["height"])
    example.context.feature["filename"].bytes_list.value.append(filename)

    object_frames = example.context.feature["object_in_frame"].int64_list.value
    for idx in context_features["object_in_frame"]:
        object_frames.append(idx)

    image_features = example.feature_lists.feature_list["images"]
    bboxes_features = example.feature_lists.feature_list["anno/bboxes"]
    occlusion_features = example.feature_lists.feature_list["anno/occluded"]
    generation_features = example.feature_lists.feature_list["anno/generated"]
    track_id_features = example.feature_lists.feature_list["anno/trackid"]
    label_feature = example.feature_lists.feature_list["anno/label"]
    for image,bboxes,occluded,generated,id,label in zip(sequence_features["images"],
                            sequence_features["anno/bboxes"],
                            sequence_features["anno/occluded"],
                            sequence_features["anno/generated"],
                            sequence_features["anno/trackid"],
                            sequence_features["anno/label"]):
        image_features.feature.add().bytes_list.value.append(image)
        add_float_list_feature(bboxes_features.feature.add(),bboxes)
        add_int64_list_feature(occlusion_features.feature.add(),occluded)
        add_int64_list_feature(generation_features.feature.add(), generated)
        add_int64_list_feature(track_id_features.feature.add(), id)
        add_int64_list_feature(label_feature.feature.add(), label)

    return example
