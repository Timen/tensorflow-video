from __future__ import absolute_import
import tensorflow as tf

slim = tf.contrib.slim

def remap_vars(variables,remove_scope):
    variables_dict = {}
    for var in variables:
        mapped_name = var.name.replace(remove_scope+"/",'')
        variables_dict[mapped_name[:-2]] = var
    return variables_dict


def init_weights(scope_name, path, ignore_strings=[]):
    if path == None:
        return

    # look for checkpoint
    model_path = tf.train.latest_checkpoint(path)
    initializer_fn = None

    if model_path:
        # only restore variables in the scope_name scope

        variables_to_restore = slim.get_variables_to_restore(include=[scope_name])
        variables_to_restore = filter(lambda variable: not any( ignore_string in variable.name for ignore_string in ignore_strings) , variables_to_restore)
        variables_to_restore = remap_vars(variables_to_restore,scope_name)

        # Create the saver which will be used to restore the variables.
        initializer_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore)
    else:
        print("could not find the fine tune ckpt at {}".format(path))
        exit()

    def InitFn(scaffold,sess):
        initializer_fn(sess)
    return InitFn