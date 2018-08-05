from __future__ import absolute_import

import tensorflow as tf
import multiprocessing

import anchors
from object_detection import preprocessor
MAX_NUM_INSTANCES = 10

def pad_to_fixed_size(data, pad_value, output_shape):
  """Pad data to a fixed length at the first dimension.
  Args:
    data: Tensor to be padded to output_shape.
    pad_value: A constant value assigned to the paddings.
    output_shape: The output shape of a 2D tensor.
  Returns:
    The Padded tensor with output_shape [max_num_instances, dimension].
  """
  max_num_instances = output_shape[0]
  dimension = output_shape[1]
  data = tf.reshape(data, [-1, dimension])
  num_instances = tf.shape(data)[0]
  tf.Assert(tf.less_equal(num_instances, max_num_instances), [num_instances])
  pad_length = max_num_instances - num_instances
  paddings = pad_value * tf.ones([pad_length, dimension])
  padded_data = tf.concat([data, paddings], axis=0)
  padded_data = tf.reshape(padded_data, output_shape)
  return padded_data


class InputProcessor(object):
  """Base class of Input processor."""

  def __init__(self, image, output_size):
    """Initializes a new `InputProcessor`.
    Args:
      image: The input image before processing.
      output_size: The output image size after calling resize_and_crop_image
        function.
    """
    self._image = image
    self._output_size = output_size
    # Parameters to control rescaling and shifting during preprocessing.
    # Image scale defines scale from original image to scaled image.
    self._image_scale = tf.constant(1.0)
    # The integer height and width of scaled image.
    self._scaled_height = tf.shape(image)[0]
    self._scaled_width = tf.shape(image)[1]
    # The x and y translation offset to crop scaled image to the output size.
    self._crop_offset_y = tf.constant(0)
    self._crop_offset_x = tf.constant(0)

  def normalize_image(self):
    """Normalize the image to zero mean and unit variance."""
    # The image normalization is identical to Cloud TPU ResNet.
    self._image = tf.image.convert_image_dtype(self._image, dtype=tf.float32)
    offset = tf.constant([123,117,104])
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    self._image -= offset


  def set_training_random_scale_factors(self, scale_min, scale_max):
    """Set the parameters for multiscale training."""
    # Select a random scale factor.
    random_scale_factor = tf.random_uniform([], scale_min, scale_max)
    scaled_size = tf.to_int32(random_scale_factor * self._output_size)

    # Recompute the accurate scale_factor using rounded scaled image size.
    height = tf.shape(self._image)[0]
    width = tf.shape(self._image)[1]
    max_image_size = tf.to_float(tf.maximum(height, width))
    image_scale = tf.to_float(scaled_size) / max_image_size

    # Select non-zero random offset (x, y) if scaled image is larger than
    # self._output_size.
    scaled_height = tf.to_int32(tf.to_float(height) * image_scale)
    scaled_width = tf.to_int32(tf.to_float(width) * image_scale)
    offset_y = tf.to_float(scaled_height - self._output_size)
    offset_x = tf.to_float(scaled_width - self._output_size)
    offset_y = tf.maximum(0.0, offset_y) * tf.random_uniform([], 0, 1)
    offset_x = tf.maximum(0.0, offset_x) * tf.random_uniform([], 0, 1)
    offset_y = tf.to_int32(offset_y)
    offset_x = tf.to_int32(offset_x)
    self._image_scale = image_scale
    self._scaled_height = scaled_height
    self._scaled_width = scaled_width
    self._crop_offset_x = offset_x
    self._crop_offset_y = offset_y

  def set_scale_factors_to_output_size(self):
    """Set the parameters to resize input image to self._output_size."""
    # Compute the scale_factor using rounded scaled image size.
    height = tf.shape(self._image)[0]
    width = tf.shape(self._image)[1]
    max_image_size = tf.to_float(tf.maximum(height, width))
    image_scale = tf.to_float(self._output_size) / max_image_size
    scaled_height = tf.to_int32(tf.to_float(height) * image_scale)
    scaled_width = tf.to_int32(tf.to_float(width) * image_scale)
    self._image_scale = image_scale
    self._scaled_height = scaled_height
    self._scaled_width = scaled_width

  def resize_and_crop_image(self, method=tf.image.ResizeMethod.BILINEAR):
    """Resize input image and crop it to the self._output dimension."""
    scaled_image = tf.image.resize_images(
        self._image, [self._scaled_height, self._scaled_width], method=method)
    scaled_image = scaled_image[
        self._crop_offset_y:self._crop_offset_y + self._output_size,
        self._crop_offset_x:self._crop_offset_x + self._output_size, :]
    output_image = tf.image.pad_to_bounding_box(
        scaled_image, 0, 0, self._output_size, self._output_size)
    return output_image


class DetectionInputProcessor(InputProcessor):
  """Input processor for object detection."""

  def __init__(self, image, output_size, boxes=None, classes=None):
    InputProcessor.__init__(self, image, output_size)
    self._boxes = boxes
    self._classes = classes

  def random_horizontal_flip(self):
    """Randomly flip input image and bounding boxes."""
    self._image, self._boxes = preprocessor.random_horizontal_flip(
        self._image, boxes=self._boxes)

  def clip_boxes(self, boxes):
    """Clip boxes to fit in an image."""
    boxes = tf.where(tf.less(boxes, 0), tf.zeros_like(boxes), boxes)
    boxes = tf.where(tf.greater(boxes, self._output_size - 1),
                     (self._output_size - 1) * tf.ones_like(boxes), boxes)
    return boxes

  def resize_and_crop_boxes(self):
    """Resize boxes and crop it to the self._output dimension."""
    boxlist = preprocessor.box_list.BoxList(self._boxes)
    boxes = preprocessor.box_list_scale(
        boxlist, self._scaled_height, self._scaled_width).get()
    # Adjust box coordinates based on the offset.
    box_offset = tf.stack([self._crop_offset_y, self._crop_offset_x,
                           self._crop_offset_y, self._crop_offset_x,])
    boxes -= tf.to_float(tf.reshape(box_offset, [1, 4]))
    # Clip the boxes.
    boxes = self.clip_boxes(boxes)
    # Filter out ground truth boxes that are all zeros.
    indices = tf.where(tf.not_equal(tf.reduce_sum(boxes, axis=1), 0))
    boxes = tf.gather_nd(boxes, indices)
    classes = tf.gather_nd(self._classes, indices)
    return boxes, classes

  @property
  def image_scale(self):
    # Return image scale from original image to scaled image.
    return self._image_scale

  @property
  def image_scale_to_original(self):
    # Return image scale from scaled image to original image.
    return 1.0 / self._image_scale

  @property
  def offset_x(self):
    return self._crop_offset_x

  @property
  def offset_y(self):
    return self._crop_offset_y
class InputReader(object):
  """Input reader for dataset."""

    def __init__(self, file_pattern, is_training,params):
        self._file_pattern = file_pattern
        self._is_training = is_training
        self._max_num_instances = MAX_NUM_INSTANCES
        input_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                        params['num_scales'],
                                        params['aspect_ratios'],
                                        params['anchor_scale'],
                                        params['image_size'])
        self.anchor_labeler = anchors.AnchorLabeler(input_anchors, params['num_classes'])


    def _parse_function(self,example_proto, image_size, num_classes,sequence_length,training,mean_value=(123,117,104),method="crop"):
        """
        Parses tf-records created with build_imagenet_data.py
        :param example_proto:
            Single example from tf record
        :param image_size:
            Output image size
        :param num_classes:
            Number of classes in dataset
        :param training:
            Whether or not the model is training
        :param mean_value:
            Imagenet mean to subtract from the output iamge
        :param method:
            How to generate the input image
        :return:
            Features dict containing image, and labels dict containing class index and one hot vector
        """

        # Define how to parse the example
        context_features = {
            "length": tf.FixedLenFeature([], dtype=tf.int64),
            # "width": tf.FixedLenFeature([], dtype=tf.int64),
            # "height": tf.FixedLenFeature([], dtype=tf.int64),
            # "filename": tf.FixedLenFeature([], dtype=tf.string),
            # "object_in_frame": tf.VarLenFeature(dtype=tf.int64)
        }
        sequence_features = {
            "images": tf.FixedLenSequenceFeature([], dtype=tf.string),
            "anno/bboxes": tf.VarLenFeature( dtype=tf.float32),
            # "anno/occluded": tf.VarLenFeature(dtype=tf.int64),
            # "anno/generated": tf.VarLenFeature(dtype=tf.int64),
            # "anno/trackid": tf.VarLenFeature(dtype=tf.int64),
            "anno/label": tf.VarLenFeature(dtype=tf.int64)
        }



        # Parse example using schema
        context_features,sequence_features = tf.parse_single_sequence_example(example_proto,context_features=context_features,sequence_features=sequence_features)

        # index = tf.cast(tf.reshape(tf.sparse_tensor_to_dense(context_features["object_in_frame"]),[-1,1]),tf.int32)

        first_frame = tf.random_uniform([1],maxval=tf.maximum(tf.cast(context_features["length"],tf.int32)-1-sequence_length,1),dtype=tf.int32)[0]
        indices = tf.range(first_frame, first_frame+sequence_length, dtype=tf.int32)
        jpeg_images = tf.gather(sequence_features["images"],indices)

        image_size = tf.cast(image_size,tf.int32)
        mean_value = tf.cast(tf.stack(mean_value),tf.float32)

        def process_fn(jpeg_image,boxes,classes):
            image = tf.image.decode_jpeg(jpeg_image)
            input_processor = DetectionInputProcessor(
                image, image_size, boxes, classes)
            input_processor.normalize_image()
            if training:
                input_processor.random_horizontal_flip()
            else:
                input_processor.set_scale_factors_to_output_size()
            image = input_processor.resize_and_crop_image()
            boxes, classes = input_processor.resize_and_crop_boxes()

            # Assign anchors.
            (cls_targets, box_targets,
             num_positives) = self.anchor_labeler.label_anchors(boxes, classes)

            # Pad groundtruth data for evaluation.
            image_scale = input_processor.image_scale_to_original
            boxes *= image_scale
            boxes = pad_to_fixed_size(boxes, -1, [self._max_num_instances, 4])
            classes = pad_to_fixed_size(classes, -1, [self._max_num_instances, 1])
            return (image, cls_targets, box_targets, num_positives,
                    image_scale, boxes,classes)

        processed_data = tf.map_fn(process_fn,jpeg_images,dtype=tf.float32)
        # subtract mean


        # subtract 1 from class index as background class 0 is not used
        label_indices = tf.cast(tf.reshape(tf.sparse_tensor_to_dense(sequence_features["anno/label"]),[-1,1]),tf.int32)
        label_indices = tf.gather(label_indices, indices)+1

        def process_labels(label_idx):
            return tf.one_hot(label_idx, num_classes)

        label_vec = tf.map_fn(process_labels,label_indices,dtype=tf.float32)
        images = tf.reshape(images,[-1,image_size,image_size,3])
        length = tf.shape(images)[0]
        loss_mask = tf.pad(tf.ones(length,tf.float32),[[0,sequence_length-length]])
        data_tuple = {"images": images}, \
               {"class_idx": tf.reshape(label_indices,[-1,1]),
                "class_vec": tf.reshape(label_vec,[-1,1,num_classes]),
               "example_length":tf.reshape(length,[1]),
                "loss_masks":loss_mask}
        return data_tuple



class ReadTFRecords(object):
    def __init__(self, image_size, batch_size, num_classes):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes

    def __call__(self, glob_pattern,sequence_length,training=True):
        """
        Read tf records matching a glob pattern
        :param glob_pattern:
            glob pattern eg. "/usr/local/share/Datasets/Imagenet/train-*.tfrecords"
        :param training:
            Whether or not to shuffle the data for training and evaluation
        :return:
            Iterator generating one example of batch size for each training step
        """
        threads = multiprocessing.cpu_count()
        with tf.name_scope("tf_record_reader"):

            # generate file list
            files = tf.data.Dataset.list_files(glob_pattern, shuffle=training)

            # parallel fetch tfrecords dataset using the file list in parallel
            dataset = files.apply(tf.contrib.data.parallel_interleave(
                lambda filename: tf.data.TFRecordDataset(filename), cycle_length=threads))



            # map the parse  function to each example individually in threads parallel calls
            dataset = dataset.map(map_func=lambda example: _parse_function(example, self.image_size, self.num_classes,sequence_length,training=training),
                                  num_parallel_calls=threads)

            # shuffle and repeat examples for better randomness and allow training beyond one epoch
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(4 * self.batch_size))


            # batch the examples
            dataset = dataset.padded_batch(batch_size=self.batch_size,padded_shapes=({"images": [sequence_length,self.image_size,self.image_size,3]},
                                                                                        {"class_idx":[sequence_length,1],
                                                                                      "class_vec":[sequence_length,1,self.num_classes],
                                                                                         "example_length":[1],
                                                                                        "loss_masks":[sequence_length]}))

            #prefetch batch
            dataset = dataset.prefetch(buffer_size=2*self.batch_size)

            return dataset.make_one_shot_iterator().get_next()
