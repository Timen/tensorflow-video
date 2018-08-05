#!/usr/bin/python
# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Converts ImageNet data to TFRecords file format with Example protos.

The raw ImageNet data set is expected to reside in JPEG files located in the
following directory structure.

  data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
  data_dir/n01440764/ILSVRC2012_val_00000543.JPEG
  ...

where 'n01440764' is the unique synset label associated with
these images.

The training data set consists of 1000 sub-directories (i.e. labels)
each containing 1200 JPEG images for a total of 1.2M JPEG images.

The evaluation data set consists of 1000 sub-directories (i.e. labels)
each containing 50 JPEG images for a total of 50K JPEG images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-01023-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128

Each validation TFRecord file contains ~390 records. Each training TFREcord
file contains ~1250 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always 'JPEG'

  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [1, 1000] where 0 is not used.
  image/class/synset: string specifying the unique ID of the label,
    e.g. 'n01440764'
  image/class/text: string specifying the human-readable version of the label
    e.g. 'red fox, Vulpes vulpes'

  image/object/bbox/xmin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/xmax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/label: integer specifying the index in a classification
    layer. The label ranges from [1, 1000] where 0 is not used. Note this is
    always identical to the image label.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.

Running this script using 16 threads may take around ~2.5 hours on an HP Z420.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import csv
import numpy as np
import process_tools
import tfrecord_tools
from collections import defaultdict
import tensorflow as tf

tf.app.flags.DEFINE_string('data_dir', '/tmp/',
                           'ILSVRC directory')

tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 16,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 16,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('max_sequence_length', 100,
                            'Maximum length of each sequence.')
tf.app.flags.DEFINE_integer('min_sequence_length', 20,
                            'Minimum length of each sequence.')

tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_integer('max_square_size', 420,
                            'Maximum size for images based on max_square_size^2')
tf.app.flags.DEFINE_integer('alternate_even_odd', 1,
                            'Alternate the sequences to either contain even, or odd frames')
tf.app.flags.DEFINE_integer('overlap_sequences', 1,
                            'Overlap sequences, by half the sequence length')
FLAGS = tf.app.flags.FLAGS


def get_files_with_extension(dir, extension):
    filenames = os.listdir(dir)
    return [os.path.splitext(filename)[0] for filename in filenames if filename.endswith(extension)]

def split_sequence(length):

    sequence_ranges = []
    if bool(FLAGS.alternate_even_odd):
        even = range(0, FLAGS.max_sequence_length, 2)
        odd = range(1, FLAGS.max_sequence_length + 1, 2)
        even_parity = True
    else:
        even_odd  = range(0, FLAGS.max_sequence_length, 1)

    if bool(FLAGS.overlap_sequences):
        start_idxes = range(0, length, int(FLAGS.max_sequence_length / 2))
    else:
        start_idxes = range(0, length, FLAGS.max_sequence_length)


    for start_idx in start_idxes:
        if bool(FLAGS.alternate_even_odd):
            if even_parity:
                sequence_cut = [x + start_idx for x in even if (x + start_idx)<length]
            else:
                sequence_cut = [x + start_idx for x in odd if (x + start_idx)<length]
            even_parity = not even_parity
        else:
            sequence_cut = [x + start_idx for x in even_odd if (x + start_idx) < length]
        if len(sequence_cut)>FLAGS.min_sequence_length:
            sequence_ranges.append(sequence_cut)
    return sequence_ranges

def _process_image_files_batch(coder, thread_index, ranges, name, filenames, directory, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      synsets: list of strings; each string is a unique WordNet ID
      labels: list of integer; each integer identifies the ground truth
      humans: list of strings; each string is a human-readable label
      bboxes: list of bounding boxes for each image. Note that each entry in this
        list might contain from 0+ entries corresponding to the number of bounding
        box annotations for the image.
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    sequences = []
    for i in range(ranges[thread_index][0],ranges[thread_index][1]):
        image_dir = os.path.join(directory, "Data/VID", name, filenames[i])
        anno_dir = os.path.join(directory, "Annotations/VID", name, filenames[i])
        frame_names = get_files_with_extension(image_dir, "JPEG")
        frame_names.sort()
        for sequence_range in  split_sequence(len(frame_names)):
            sequence = {"image_dir": image_dir, "anno_dir": anno_dir, "range":sequence_range, "frame_names":frame_names}
            sequences.append(sequence)
    shard_ranges = np.linspace(0,
                               len(sequences),
                               num_shards_per_batch + 1).astype(int)
    num_sequences_in_thread = len(sequences)
    random.shuffle(sequences)

    shard_counter = 0
        # for sequence_range in split_sequence(len(frame_names)):
    for s in range(num_shards_per_batch):
        counter = 0
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d.tfrecord' % (name, shard+1, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        sequences_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in sequences_in_shard:
            sequence = sequences[i]
            image_dir = sequence["image_dir"]
            anno_dir = sequence["anno_dir"]
            frame_names = sequence["frame_names"]
            sequence_range = sequence["range"]
            sequence_features = defaultdict(list)
            object_in_frame_indices = []

            for idx,frame_idx in enumerate(sequence_range):
                image_path = os.path.join(image_dir, frame_names[frame_idx] + ".JPEG")
                anno_path = os.path.join(anno_dir, frame_names[frame_idx] + ".xml")
                image_buffer, height, width = process_tools.process_image(image_path, coder)
                sequence_features["images"].append(image_buffer)
                annotated_objects = process_tools.process_xml(anno_path)
                if len(annotated_objects) > 0:
                    boxes = []
                    ids = []
                    generations = []
                    occlusions = []
                    labels = []

                    for annotated_object in annotated_objects:
                        boxes.append(annotated_object["bbox"])
                        occlusions.append(annotated_object["occluded"])
                        generations.append(annotated_object["generated"])
                        ids.append(annotated_object["id"])
                        labels.append(annotated_object["label"])
                    sequence_features["anno/bboxes"].append(boxes)
                    sequence_features["anno/occluded"].append(occlusions)
                    sequence_features["anno/generated"].append(generations)
                    sequence_features["anno/trackid"].append(ids)
                    sequence_features["anno/label"].append(labels)
                    object_in_frame_indices.append(idx)
                else:
                    sequence_features["anno/bboxes"].append([])
                    sequence_features["anno/occluded"].append([])
                    sequence_features["anno/generated"].append([])
                    sequence_features["anno/trackid"].append([])
                    sequence_features["anno/label"].append([])
            if len(object_in_frame_indices) == 0:
                print("%s [thread %d]:No objects in sequence"%(datetime.now(), thread_index))
                sys.stdout.flush()
                continue
            context_features = {}
            context_features["width"] = width
            context_features["height"] = height
            context_features["length"] = idx+1
            context_features["object_in_frame"] = object_in_frame_indices
            example = tfrecord_tools.convert_to_example(image_dir, sequence_features, context_features)
            writer.write(example.SerializeToString())
            counter += 1
            shard_counter += 1


            if not shard_counter % 100:
                print('%s [thread %d]: Processed %d of %d sequences in thread batch.' %
                      (datetime.now(), thread_index, shard_counter, num_sequences_in_thread))
                sys.stdout.flush()
        writer.close()
        print('%s [thread %d]: Wrote %d sequences to %s' %
              (datetime.now(), thread_index, counter, output_file))
        sys.stdout.flush()
    print('%s [thread %d]: Wrote %d sequences to %d shards.' %
          (datetime.now(), thread_index, shard_counter , num_shards_per_batch))
    sys.stdout.flush()


def _process_image_files(name, filenames, data_dir, num_shards):
    """Process and save list of images as TFRecord of Example protos.

    Args:
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      synsets: list of strings; each string is a unique WordNet ID
      labels: list of integer; each integer identifies the ground truth
      humans: list of strings; each string is a human-readable label
      bboxes: list of bounding boxes for each image. Note that each entry in this
        list might contain from 0+ entries corresponding to the number of bounding
        box annotations for the image.
      num_shards: integer number of shards for this data set.
    """

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = process_tools.ImageCoder((FLAGS.max_square_size,FLAGS.max_square_size))

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames, data_dir, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d videos in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir, name):
    """Build a list of all images files and labels in the data set.

    Args:
      data_dir: string, path to the root directory of images.

        Assumes that the ImageNet data set resides in JPEG files located in
        the following directory structure.

          data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
          data_dir/n01440764/ILSVRC2012_val_00000543.JPEG

        where 'n01440764' is the unique synset label associated with these images.

      labels_file: string, path to the labels file.

        The list of valid labels are held in this file. Assumes that the file
        contains entries as such:
          n01440764
          n01443537
          n01484850
        where each line corresponds to a label expressed as a synset. We map
        each synset contained in the file to an integer (based on the alphabetical
        ordering) starting with the integer 1 corresponding to the synset
        contained in the first line.

        The reason we start the integer labels at 1 is to reserve label 0 as an
        unused background class.

    Returns:
      filenames: list of strings; each string is a path to an image file.
      synsets: list of strings; each string is a unique WordNet ID.
      labels: list of integer; each integer identifies the ground truth.
    """
    print('Determining list of input files and labels from %s.' % data_dir)
    filenames = []
    total_files = 0

    set_txt = os.path.join(data_dir, "ImageSets", "VID", name + "_all.txt")
    with open(set_txt) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        for row in csv_reader:
            filenames.append(str(row[0]))
            total_files = total_files + 1
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    random.seed(12345)
    random.shuffle(filenames)

    return filenames


def _process_dataset(name, data_dir, num_shards):
    """Process a complete data set and save it as a TFRecord.

    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
      synset_to_human: dict of synset to human labels, e.g.,
        'n02119022' --> 'red fox, Vulpes vulpes'
      image_to_bboxes: dictionary mapping image file names to a list of
        bounding boxes. This list contains 0+ bounding boxes.
    """
    filenames = _find_image_files(data_dir, name)
    _process_image_files(name, filenames, data_dir, num_shards)



def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.validation_shards % FLAGS.num_threads, (
        'Please make the FLAGS.num_threads commensurate with '
        'FLAGS.validation_shards')
    print('Saving results to %s' % FLAGS.output_directory)
    # # Run it!
    # _process_dataset('val', FLAGS.data_dir,
    #                  FLAGS.validation_shards)
    # exit()
    _process_dataset('train', FLAGS.data_dir,
                     FLAGS.train_shards)


if __name__ == '__main__':
    tf.app.run()
