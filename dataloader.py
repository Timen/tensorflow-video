from __future__ import absolute_import

import tensorflow as tf
import multiprocessing

def _parse_function(example_proto, image_size, num_classes,sequence_length,training,mean_value=(123,117,104),method="crop"):
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
        "anno/trackid": tf.VarLenFeature(dtype=tf.int64),
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

    def process_fn(jpeg_image):
        image = tf.image.decode_jpeg(jpeg_image)
        image = tf.image.resize_images(image, [image_size, image_size])
        image = image - mean_value
        return image

    images = tf.map_fn(process_fn,jpeg_images,dtype=tf.float32)
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
