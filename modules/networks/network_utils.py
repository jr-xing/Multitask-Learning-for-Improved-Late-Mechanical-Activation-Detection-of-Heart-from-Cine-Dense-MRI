# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 13:46:51 2021

@author: Jerry Xing
"""

# import torch
# import numpy as np
from torch import nn

def get_last_conv_channels_num(layers: list):
    last_conv_layer = [layer for layer in layers if type(layer) is nn.Conv2d][-1]
    return last_conv_layer.out_channels
    

def get_pooling_layer_num(layers: list):
    pooling_layers = len([layer for layer in layers if type(layer) is nn.MaxPool2d])
    return len(pooling_layers)