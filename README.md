## Tensorflow Video
This repository contains some examples of how to do training on video sequences using tensorflow. In contrast to a lot of methods
that train for video classification or other applications, the feature extractor can also be trained during training in the 
style of an RNN. This repo contains the whole pipeline from loading videos into tfrecords, to masking the loss and optimizing
the Dynamic RNN by passing the sequence lengths.
