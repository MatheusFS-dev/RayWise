There are some options for the NN.

1. Working with two separated branches, one for the lidar and another for the coords. In the coords we could use an increase, stay, decrease logic. Or an increasing.
2. Another option is to preprocess the coords, apply a normalization and concatenate with the lidar features, like 20, 200, 12.
then use cnn