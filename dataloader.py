from __future__ import absolute_import

import tensorflow as tf
import multiprocessing

import anchors
from object_detection import preprocessor
from collections import OrderedDict
import glob
MAX_NUM_INSTANCES = 50


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


    def normalize_image(self):
        """Normalize the image to zero mean and unit variance."""
        # The image normalization is identical to Cloud TPU ResNet.
        self._image = tf.image.convert_image_dtype(self._image, dtype=tf.float32)
        offset = tf.constant([123, 117, 104], dtype=tf.float32)
        self._image -= offset

    def resize_and_crop_image(self, method=tf.image.ResizeMethod.BILINEAR):
        """Resize input image and crop it to the self._output dimension."""
        scaled_image = tf.image.resize_images(
            self._image, [self._output_size, self._output_size], method=method)
        return scaled_image


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
            boxlist, self._output_size, self._output_size).get()

        # Clip the boxes.
        boxes = self.clip_boxes(boxes)

        # Filter out ground truth boxes that are all zeros.
        indices = tf.where(tf.not_equal(tf.reduce_sum(boxes, axis=1), 0))
        boxes = tf.gather_nd(boxes, indices)
        classes = tf.gather_nd(self._classes, indices)
        return boxes, classes



class InputReader(object):
    """Input reader for dataset."""

    def __init__(self, params):
        self._max_num_instances = MAX_NUM_INSTANCES
        self._image_size = params["image_size"]
        self._num_classes = params["num_classes"]
        input_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                        params['num_scales'],
                                        params['aspect_ratios'],
                                        params['anchor_scale'],
                                        (params['image_size'] - 5))
        self.anchor_labeler = anchors.AnchorLabeler(input_anchors, params['num_classes'])

    def parse_function(self, example_proto, sequence_length, training):
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
            "anno/bboxes": tf.VarLenFeature(dtype=tf.float32),
            # "anno/occluded": tf.VarLenFeature(dtype=tf.int64),
            # "anno/generated": tf.VarLenFeature(dtype=tf.int64),
            # "anno/trackid": tf.VarLenFeature(dtype=tf.int64),
            "anno/label": tf.VarLenFeature(dtype=tf.int64)
        }

        # Parse example using schema
        context_features, sequence_features = tf.parse_single_sequence_example(example_proto,
                                                                               context_features=context_features,
                                                                               sequence_features=sequence_features)

        # index = tf.cast(tf.reshape(tf.sparse_tensor_to_dense(context_features["object_in_frame"]),[-1,1]),tf.int32)
        example_length = tf.cast(context_features["length"], tf.int32)
        first_frame = tf.random_uniform([1], maxval=tf.maximum(
            example_length - sequence_length, 1), dtype=tf.int32)[0]
        indices = tf.range(first_frame, tf.minimum(first_frame + sequence_length,example_length), dtype=tf.int32)
        jpeg_images = tf.gather(sequence_features["images"], indices)

        indices = tf.cast(indices,tf.int64)
        bboxes = tf.sparse_tensor_to_dense(sequence_features["anno/bboxes"])
        bboxes = tf.gather(bboxes, indices)

        classes = tf.sparse_tensor_to_dense(sequence_features["anno/label"])
        classes_select = tf.cast(tf.gather(classes, indices),tf.float32)

        def process_fn(inputs):
            jpeg_image, boxes,classes = inputs

            boxes = tf.reshape(boxes, [-1, 4])
            classes = tf.reshape(classes, [-1, 1])
            mask = tf.greater(classes[:,0],0)
            boxes = tf.boolean_mask(boxes,mask)
            classes = tf.boolean_mask(classes, mask)-1

            image = tf.cast(tf.image.decode_jpeg(jpeg_image), tf.float32)
            input_processor = DetectionInputProcessor(
                image, self._image_size, boxes, classes)
            input_processor.normalize_image()

            image = input_processor.resize_and_crop_image()
            boxes, classes = input_processor.resize_and_crop_boxes()
            # Assign anchors.
            (cls_targets, box_targets,
             num_positives) = self.anchor_labeler.label_anchors(boxes, classes)

            boxes = pad_to_fixed_size(boxes, -1, [self._max_num_instances, 4])
            classes = pad_to_fixed_size(classes, -1, [self._max_num_instances, 1])

            return (image, cls_targets, box_targets, num_positives, boxes, classes)


        processed_data = tf.map_fn(process_fn, (jpeg_images, bboxes,classes_select), dtype=(tf.float32,
                                                                                      OrderedDict({3: tf.int32,
                                                                                                   4: tf.int32,
                                                                                                   5: tf.int32}),
                                                                                      OrderedDict(
                                                                                          {3: tf.float32,
                                                                                           4: tf.float32,
                                                                                           5: tf.float32}),
                                                                                      tf.float32,
                                                                                      tf.float32,
                                                                                      tf.float32))
        # subtract mean

        images = tf.reshape(processed_data[0], [-1, self._image_size, self._image_size, 3])
        length = tf.shape(images)[0]
        loss_mask = tf.pad(tf.ones(length, tf.float32), [[0, sequence_length - length]])

        data_tuple = ({"images": images,
                      "example_length": tf.reshape(length, [1])}, \
                     {"cls_targets": processed_data[1],
                      "box_targets": processed_data[2],
                      "num_positives": processed_data[3],
                      "boxes":processed_data[4],
                      "classes":processed_data[5],
                      "loss_masks": loss_mask})
        self.store_shapes(data_tuple[1],sequence_length)
        return data_tuple

    def store_shapes(self,labels,sequence_length):
        shape_dict = {}
        for key,value in labels.iteritems():
            if isinstance(value, dict):
                nested_shape_dict = OrderedDict()
                for nested_key, tensor in value.iteritems():
                    nested_shape_dict[nested_key] = [sequence_length]+tensor.get_shape().as_list()[1:]
                shape_dict[key] = nested_shape_dict
            else:
                shape_dict[key] = [sequence_length]+value.get_shape().as_list()[1:]
        self.shapes_dict = shape_dict



class ReadTFRecords(object):
    def __init__(self, batch_size, params):
        self.batch_size = batch_size
        self.params = params
        self.image_size = params["image_size"]

    def __call__(self, glob_pattern, sequence_length, training=True):
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
            input_class = InputReader(self.params)
            # map the parse  function to each example individually in threads parallel calls
            dataset = dataset.map(
                map_func=lambda example: input_class.parse_function(example, sequence_length,
                                                                     training),
                num_parallel_calls=threads)

            # shuffle and repeat examples for better randomness and allow training beyond one epoch
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(4 * self.batch_size))

            # batch the examples
            dataset = dataset.padded_batch(batch_size=self.batch_size, padded_shapes=(
                {"images": [sequence_length, self.image_size, self.image_size, 3],
                 "example_length": [1]},
                input_class.shapes_dict))

            # prefetch batch
            dataset = dataset.prefetch(buffer_size=sequence_length)

            return dataset.make_one_shot_iterator().get_next()

    def test(self,glob_pattern, sequence_length,params, training=True):
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer(glob.glob(glob_pattern), num_epochs=1)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        input_class = InputReader(params)
        features,labels = input_class.parse_function(serialized_example,sequence_length,training)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)
            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(0,1000):
                print(i)
                print(sess.run([features,labels]))

