from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels=3, kernel_size=3, number_filters=3, stride=1, padding=0, name="conv"),
            MaxPoolingLayer(pool_size=2, stride=2, name="maxpool"),
            flatten(name="flat"),
            fc(27, 5, 0.02, name="fc")
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels=3, kernel_size=4, number_filters=16, stride=2, padding=0, name="conv1"),
            MaxPoolingLayer(pool_size=4, stride=1, name="maxpool1"),
            ConvLayer2D(input_channels=16, kernel_size=4, number_filters=16, stride=2, padding=0, name="conv2"),
            MaxPoolingLayer(pool_size=2, stride=1, name="maxpool2"),
            flatten(name="flat"),
            fc(256, 10, 0.02, name="fc")            
            ########### END ###########
        )
