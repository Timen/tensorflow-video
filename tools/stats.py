from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.framework import ops
from collections import defaultdict
import pandas as pd
import os
supported_stat_ops = ["Conv2D", "MatMul", "VariableV2", "MaxPool","AvgPool","Add"]
exclude_in_name = ["gradients", "Initializer", "Regularizer", "AssignMovingAvg", "Momentum", "BatchNorm"]


class ModelStats(tf.train.SessionRunHook):
    """Logs model stats to a csv."""

    def __init__(self, path,normalize_factor, min_flops=32000):
        """
        Set class variables
        :param scope_name:
            Used to filter for tensors which name contain that specific variable scope
        :param path:
            path to model dir
        :param batch_size:
            batch size during training
        """
        self.normalize_factor = normalize_factor
        self.path = path
        self.min_flops = min_flops
    def begin(self):
        """
            Method to output statistics of the model to an easy to read csv, listing the multiply accumulates(maccs) and
            number of parameters, in the model dir.
        :param session:
            Tensorflow session
        :param coord:
            unused
        """
        # get graph and operations
        graph = tf.profiler.profile(
            tf.get_default_graph(),
            options=tf.profiler.ProfileOptionBuilder(
                tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()).float_operation())
        # setup dictionaries
        stat_dict = {}
        # iterate over tensors
        for tensor in graph.children:
            name = tensor.name

            # check is scope_name is in name, or any of the excluded strings
            if any(exclude_name in name for exclude_name in exclude_in_name):
                continue
            # Check if type is considered for the macc calculation
            if not any(supported_op in name for supported_op in supported_stat_ops):
                continue
            if (tensor.total_float_ops/self.normalize_factor) > self.min_flops:
                stat_dict[tensor.name] = tensor.total_float_ops/self.normalize_factor


        df = pd.DataFrame.from_dict(stat_dict, orient='index')
        df.loc['Total'] = df[0].sum()
        df = df.sort_values(by=[0], ascending=False)
        df.to_csv(os.path.join(self.path,'model_stats.csv'))
