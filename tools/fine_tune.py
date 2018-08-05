from __future__ import absolute_import
import tensorflow as tf

slim = tf.contrib.slim

def remap_vars(variables,remove_scopes):
    variables_dict = {}
    for var in variables:
        mapped_name = "/".join(var.name.split("/")[remove_scopes:])
        variables_dict[mapped_name[:-2]] = var
    return variables_dict


def init_weights(scope_name, path, ignore_vars=[]):
    if path == None:
        return

    # look for checkpoint
    model_path = tf.train.latest_checkpoint(path)
    initializer_fn = None

    if model_path:
        # only restore variables in the scope_name scope
        ignore_vars = [scope_name+var for var in ignore_vars]

        variables_to_restore = slim.get_variables_to_restore(include=[scope_name],exclude=ignore_vars)
        variables_to_restore = remap_vars(variables_to_restore,2)

        # Create the saver which will be used to restore the variables.
        initializer_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore)
    else:
        print("could not find the fine tune ckpt at {}".format(path))
        exit()

    def InitFn(scaffold,sess):
        initializer_fn(sess)
    return InitFn