from __future__ import absolute_import

# Some parameters will be divided or multiplied by 4 to compensate for the reduced batch size
# as the original used batch size 1024 and the gtx1080ti can only fit a batch size 256.

training_params = {
    # the base learning rate used in the polynomial decay
    "base_lr": 0.025,

    # how many steps to warmup the learning rate for
    "warmup_iter": 4000,

    # What learning rate to start with in the warmup phase (ramps up to base_lr)
    "warmup_start_lr": 0.001,

    # input size
    "image_size": 229,

    # Block defs each tuple(x,y,z) describes one block with x number of filters at it's largest depth
    # y number of repeated units or bottlenecks, z stride for the first unit of the block.
    "block_defs": [(32, 6, 1), (64, 6, 2), (128, 8, 2), (256, 1, 2)],

    # definition of filters, kernel size and stride of the input convolution
    "input_def": (64, (7, 7), 2),

    # number of output classes
    "num_classes": 30,

    # How many groups to use for the grouped convolutions
    "groups": 1,

    # Whether to do relu before addition of the network and the residual
    "seperate_relus": 0,
    "min_level": 3,
    "max_level": 5,
    "num_scales": 2,
    "aspect_ratios": [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
    "anchor_scale": 3.0,
    "num_filters": 64,
    "alpha":0.25,
    "gamma":1.5,
    "delta":0.1,
    "box_loss_weight":50

}
