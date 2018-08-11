from __future__ import absolute_import

import tensorflow as tf

slim = tf.contrib.slim
metrics = tf.contrib.metrics
import squeezenext_architecture as squeezenext
import tensorflow_extentions as tfe
from optimizer import PolyOptimizer
from dataloader import ReadTFRecords
import tools
import os
import model_heads
from loss import detection_loss

metrics = tf.contrib.metrics


class Model(object):
    def __init__(self, config):
        self.image_size = config["image_size"]
        self.num_classes = config["num_classes"]
        self._config = config

    def define_batch_size(self, features, labels, batch_size):
        """
        Define batch size of dictionary
        :param features:
            Feature dict
        :param labels:
            Labels dict
        :return:
            (features,label)
        """
        features = tools.define_first_dim(features, batch_size)
        labels = tools.define_first_dim(labels, batch_size)
        return (features, labels)

    def input_fn(self, file_pattern, training, batch_size, sequence_length):
        """
        Input fn of model
        :param file_pattern:
            Glob file pattern
        :param training:
            Whether or not the model is training
        :return:
            Input generator
        """
        read_tf_records = ReadTFRecords(batch_size, self._config)
        return self.define_batch_size(*read_tf_records(file_pattern, sequence_length, training=training),
                                      batch_size=batch_size)

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
        batch_size, sequence_length = tuple(labels["loss_masks"].get_shape().as_list())
        unpadded_features = features["example_length"][:, 0]


        classifier = squeezenext.SqueezeNext(self.num_classes, params["block_defs"], params["input_def"],
                                             params["groups"], params["seperate_relus"])
        with slim.arg_scope(classifier.model_arg_scope(training)):
            with tf.variable_scope("classifier"):
                classifier_endpoints,scope_string = tfe.time_distributed(features["images"], unpadded_features, classifier,
                                                            [[28, 28, 64], [14, 14, 128], [7, 7, 256]],
                                                            endpoints=["block_1/unit_0", "block_2/unit_0", "block_3/unit_0"],
                                                            return_scope_string=True)
        conv_lstm = tfe.ConvolutionalLstm(classifier_endpoints)

        with tf.variable_scope("bottleneck_lstm"):
            initializer = tfe.initial_state.make_gaussian_state_initializer(tfe.initial_state.make_variable_state_initializer(),
                                                          tf.constant(False))

            init_state = tfe.initial_state.get_initial_cell_state(conv_lstm, initializer, batch_size, tf.float32)
            predictions, last_states = tf.nn.dynamic_rnn(
                cell=conv_lstm,
                dtype=tf.float32,
                sequence_length=unpadded_features,
                inputs=classifier_endpoints,
                initial_state=init_state)

        model_head = model_heads.ModelHead(params)
        with slim.arg_scope(model_head.model_arg_scope(training)):
            with tf.variable_scope("model_head"):
                predictions = tfe.time_distributed(predictions, unpadded_features,
                                                            model_head,
                                                            model_head.output_size(predictions))

        loss, cls_loss, box_loss = detection_loss(predictions, labels, params)

        # create histogram of class spread
        tf.summary.histogram("classes", labels["cls_targets"][params["min_level"]])

        if training:
            tf.summary.scalar("box_loss", box_loss)
            tf.summary.scalar("cls_loss", cls_loss)
            # init poly optimizer
            optimizer = PolyOptimizer(params)
            # define train op
            train_op = optimizer.optimize(loss, training, params["total_steps"])

            # if params["output_train_images"] is true output images during training
            if params["output_train_images"]:
                tools.draw_box_predictions(features["images"], predictions, labels, params, sequence_length)

            # stats_hook = tools.stats.ModelStats("rnn/"+conv_lstm.scope_name+"/squeezenext", params["model_dir"],
            # batch_size*sequence_length)
            # setup fine tune scaffold
            scaffold = tf.train.Scaffold(init_op=None,
                                         init_fn=tools.fine_tune.init_weights(
                                             scope_string, params["fine_tune_ckpt"],
                                             ignore_vars=["/squeezenext/fully_connected/weights"],
                                                ignore_strings = ["RMSProp"]))

            # create estimator training spec, which also outputs the model_stats of the model to params["model_dir"]
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, scaffold=scaffold)

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_metric = tools.coco_metrics.EvaluationMetric()
            coco_metrics = eval_metric.estimator_metric_fn(tools.eval_predictions(predictions,params),tools.eval_labels(labels))

            # Define the metrics:
            # metrics_dict = {
            #     'Recall@1': tf.metrics.accuracy(tf.argmax(predictions, axis=-1), labels["class_idx"][:, 0]),
            #     'Recall@5': metrics.streaming_sparse_recall_at_k(predictions, tf.cast(labels["class_idx"], tf.int64),
            #                                                      5)
            # }
            # output eval images
            eval_summary_hook = tf.train.SummarySaverHook(
                save_steps=100,
                output_dir=os.path.join(params["model_dir"],"eval"),
                summary_op=tf.summary.image("validation", features["images"][0,:,:,:,:]))

            #return eval spec
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=coco_metrics,
                evaluation_hooks=[eval_summary_hook])
