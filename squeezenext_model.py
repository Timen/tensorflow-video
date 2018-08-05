from __future__ import absolute_import

import tensorflow as tf

slim = tf.contrib.slim
metrics = tf.contrib.metrics
import squeezenext_architecture as squeezenext
import tensorflow_extentions as tfe
from optimizer  import PolyOptimizer
from dataloader import ReadTFRecords
import tools
import os
metrics = tf.contrib.metrics

class Model(object):
    def __init__(self, config, batch_size,sequence_length):
        self.image_size = config["image_size"]
        self.num_classes = config["num_classes"]+1
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.read_tf_records = ReadTFRecords(self.image_size, self.batch_size, self.num_classes)

    def define_batch_size(self, features, labels):
        """
        Define batch size of dictionary
        :param features:
            Feature dict
        :param labels:
            Labels dict
        :return:
            (features,label)
        """
        features = tools.define_first_dim(features, self.batch_size)
        labels = tools.define_first_dim(labels, self.batch_size)
        return (features, labels)

    def input_fn(self, file_pattern,training):
        """
        Input fn of model
        :param file_pattern:
            Glob file pattern
        :param training:
            Whether or not the model is training
        :return:
            Input generator
        """
        return self.define_batch_size(*self.read_tf_records(file_pattern,self.sequence_length,training=training))

    def model_fn(self, features, labels, mode, params):
        """
        Function to create squeezenext model and setup training environment
        :param features:
            Feature dict from estimators input fn
        :param labels:
            Label dict from estimators input fn
        :param mode:
            What mode the model is in tf.estimator.ModeKeys
        :param params:
            Dictionary of parameters used to configurate the network
        :return:
            Train op, predictions, or eval op depening on mode
        """

        training = mode == tf.estimator.ModeKeys.TRAIN
        # init model class
        model = squeezenext.SqueezeNext(self.num_classes, params["block_defs"], params["input_def"], params["groups"],params["seperate_relus"])
        conv_lstm = tfe.ConvolutionalLstm( model, training,self.num_classes)
        predictions, last_states = tf.nn.dynamic_rnn(
            cell=conv_lstm,
            dtype=tf.float32,
            sequence_length=labels["example_length"][:,0],
            inputs=features["images"])
        pred_flat = tf.reshape(predictions,[-1,self.num_classes])
        label_flat = tf.reshape(labels["class_vec"],[-1,self.num_classes])
        mask_flat = tf.reshape(labels["loss_masks"],[-1])

        loss =  tf.losses.softmax_cross_entropy(label_flat, pred_flat)
        masked_losses = loss * mask_flat
        masked_losses = tf.reshape(masked_losses, [self.batch_size,self.sequence_length])
        mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / tf.cast(labels["example_length"],tf.float32)
        loss = tf.reduce_mean(mean_loss_by_example)


        # create histogram of class spread
        tf.summary.histogram("classes",labels["class_idx"])

        if training:
            # init poly optimizer
            optimizer = PolyOptimizer(params)
            # define train op
            train_op = optimizer.optimize(loss, training, params["total_steps"])

            # if params["output_train_images"] is true output images during training
            if params["output_train_images"]:
                tf.summary.image("training", features["images"][0,:,:,:,:])
            stats_hook = tools.stats.ModelStats("rnn/"+conv_lstm.scope_name+"/squeezenext", params["model_dir"],self.batch_size)
            # setup fine tune scaffold
            scaffold = tf.train.Scaffold(init_op=None,
                                         init_fn=tools.fine_tune.init_weights("rnn/"+conv_lstm.scope_name+"/squeezenext", params["fine_tune_ckpt"],ignore_vars=["/fully_connected/weights"]))

            # create estimator training spec, which also outputs the model_stats of the model to params["model_dir"]
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[stats_hook],scaffold=scaffold)



        if mode == tf.estimator.ModeKeys.EVAL:
            # Define the metrics:
            metrics_dict = {
                'Recall@1': tf.metrics.accuracy(tf.argmax(predictions, axis=-1), labels["class_idx"][:, 0]),
                'Recall@5': metrics.streaming_sparse_recall_at_k(predictions, tf.cast(labels["class_idx"], tf.int64),
                                                                 5)
            }
            # output eval images
            eval_summary_hook = tf.train.SummarySaverHook(
                save_steps=100,
                output_dir=os.path.join(params["model_dir"],"eval"),
                summary_op=tf.summary.image("validation", features["images"][0,:,:,:,:]))

            #return eval spec
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics_dict,
                evaluation_hooks=[eval_summary_hook])
