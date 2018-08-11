from __future__ import absolute_import
import tensorflow as tf
from configs import configs
from video_retinanet import Model
import argparse
from tensorflow.python import debug as tf_debug
from dataloader import ReadTFRecords

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='Training parser')
parser.add_argument('--model_dir', type=str, required=True,
                    help='Location of model_dir')
parser.add_argument('--configuration', type=str, default="v_1_0_SqNxt_23",
                    help='Name of model config file')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size during training')
parser.add_argument('--num_examples_per_epoch', type=int, default=21000*50,
                    help='Number of examples in one epoch')
parser.add_argument('--num_eval_examples', type=int, default=4000,
                    help='Number of examples in one eval epoch')
parser.add_argument('--num_epochs_per_length', type=int, default=5,
                    help='Number of epochs for training')
parser.add_argument('--num_sequence_lengths', type=int, default=5,
                    help='Number of different sequence lengths')
parser.add_argument('--training_file_pattern', type=str, required=True,
                    help='Glob for training tf records')
parser.add_argument('--validation_file_pattern', type=str, required=True,
                    help='Glob for validation tf records')
parser.add_argument('--eval_every_n_secs', type=int, default=3600,
                    help='Run eval every N seconds, default is every  hour')
parser.add_argument('--output_train_images', type=bool, default=True,
                    help='Whether to save image summary during training (Warning: can lead to large event file sizes).')
parser.add_argument('--fine_tune_ckpt', type=str, default=None,
                    help='Ckpt used for initializing the variables')
parser.add_argument('--sequence_length', type=int, default=2,
                    help='Length of each video sequence during training')
parser.add_argument('--debug', type=bool, default=False,
                    help='Whether to test the input function outside the estiamtor')
args = parser.parse_args()


def main(argv):
    """
    Main function to start training
    :param argv:
        not used
    :return:
        None
    """
    del argv  # not used

    # calculate steps per epoch
    steps_per_epoch = (args.num_examples_per_epoch / args.batch_size / args.sequence_length)
    # setup config dictionary
    config = configs[args.configuration]
    config["model_dir"] = args.model_dir
    config["output_train_images"] = args.output_train_images
    config["total_steps"] = steps_per_epoch * args.num_epochs_per_length * args.num_sequence_lengths
    config["model_dir"] = args.model_dir
    config["fine_tune_ckpt"] = args.fine_tune_ckpt
    if args.debug:
        read_tf_records = ReadTFRecords(args.batch_size, config)
        read_tf_records.test(args.training_file_pattern,args.sequence_length,config,True)
        hooks = [tf_debug.LocalCLIDebugHook()]
    else:
        hooks = None

    # init model class
    model = Model(config)
    classifier = tf.estimator.Estimator(
        model_dir=args.model_dir,
        model_fn=model.model_fn,
        params=config)


    for i in range(0,args.num_sequence_lengths):

        batch_size =  args.batch_size / (2**i)
        sequence_length = args.sequence_length * (2**i)
        steps_per_epoch = (args.num_examples_per_epoch / batch_size / sequence_length)
        total_steps = (steps_per_epoch * args.num_epochs_per_length) * (i + 1)
        # create classifier

        tf.logging.info("Total steps = {}, num_epochs = {}, batch size = {}, sequence length = {}".format(total_steps, args.num_epochs_per_length,
                                                                                    batch_size,sequence_length))


        # setup train spec
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: model.input_fn(args.training_file_pattern, True,batch_size,sequence_length),
                                            max_steps=total_steps,
                                            hooks=hooks)

        # setup eval spec evaluating ever n seconds
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: model.input_fn(args.validation_file_pattern, False,batch_size,sequence_length),
            steps=args.num_eval_examples / batch_size / sequence_length,
            throttle_secs=args.eval_every_n_secs)

        # run train and evaluate
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    classifier.evaluate(input_fn=lambda: model.input_fn(args.validation_file_pattern, False,batch_size,sequence_length),
        steps=args.num_eval_examples / args.batch_size / args.sequence_length)




if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
