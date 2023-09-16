# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 13:45:10 2021

@author: Jerry Xing
"""
import torch
import numpy as np
from torch import nn
from icecream import ic
from torch.cuda.amp import autocast

def conv2d_net(input_shape: list or tuple or int, 
             n_conv_layers=3,
             n_output_channels:int or None=None,
             conv_size=3,
             stride_size=1, 
             padding_size=1, 
             pooling_method=None, 
             pooling_kernel_size=2,
             pooling_layers_max_num=None,
             activation_func=torch.nn.ReLU(), 
             batchnorm=False, 
             batchnorm_at_end=False, activation_at_end=False):
    # https://stackoverflow.com/questions/42015156/the-order-of-pooling-and-normalization-layer-in-convnet
    # pooling -> batch normalizaion
    if n_conv_layers <=0:
        return {'layers': [], 'output_shape': input_shape}
    
    # Input shape could be:
        # 1) (C, H, W)
        # 2) (C, None, None)
        # 3) C
    # ic('------------------------')
    # ic('------------------------')
    if type(input_shape) in [list, tuple]:
        if len(input_shape) == 4:
            _, input_channel_num, input_H, input_W = input_shape
        elif len(input_shape) == 3:
            input_channel_num, input_H, input_W = input_shape
        elif len(input_shape) == 2:
            raise ValueError('Unsupported intput shape: ', input_shape)
        elif len(input_shape) == 1:
            input_channel_num = input_shape[0]
            input_H, input_W = None, None
    
    
    
    # Make sure all variables are int
    # ic(input_shape)
    # ic(n_output_channels)
    # ic(n_conv_layers)
    # ic(pooling_layers_max_num)
    # ic(input_channel_num)
    # input_H = int(input_H)
    # input_W = int(input_W)
    
    actual_pooling_layer_num = 0
    # Determine pooling settings
    if pooling_method is not None:
        if pooling_layers_max_num is None:
            # if contain pooling layers and pooling number is not specified, set number equals to number of conv layers
            pooling_layers_max_num = n_conv_layers
    else:
        pooling_layers_max_num = 0
    
    # Check if there are too much pooling
    if pooling_method is not None:
        if pooling_kernel_size ** pooling_layers_max_num > max(input_H, input_W):
            raise ValueError('Too much pooling layers!')
    
    # print('n_output_channels', n_output_channels)
    # Determine channel number of each layer according to specified output channel number or pooling method
    if n_output_channels is not None:
        # if output channel num is specified, linear interpolated inner layer channel numbers
        conv_input_channels = np.linspace(input_channel_num, n_output_channels, n_conv_layers + 1).astype(int)#[:-1]
    else:        
        # ic(pooling_method)
        # ic(n_output_channels)
        if pooling_method is None or n_output_channels == 0:
            # if output channel num is not specified and don't use pooling, keep channel number unchanged
            conv_input_channels = [input_channel_num] * (n_conv_layers + 1) # +1 to include the output of last layer
        else:
            # print('ELSE!')
            # print('pooling_layers_max_num', pooling_layers_max_num)
            # print('input_channel_num', input_channel_num)
            # if output channel num is not specified and use pooling, the channel numbers are decided by doubling
            # ic()
            conv_input_channels = []
            for layer_idx in range(pooling_layers_max_num):
                # print(layer_idx)
                conv_input_channels.append(input_channel_num * 2**layer_idx)
            conv_input_channels += [int(input_channel_num * 2**(max(pooling_layers_max_num-1, 0)))] * (n_conv_layers - pooling_layers_max_num + 1)        
            # print('conv_input_channels', conv_input_channels)
            # ic(n_conv_layers - pooling_layers_max_num + 1)
    
    # Setting layers
    # print('conv_input_channels', conv_input_channels)
    layers = []
    for layer_idx in range(n_conv_layers):
        # print('layer_idx', layer_idx)
        # Conv
        
        layers.append(
                nn.Conv2d(in_channels=conv_input_channels[layer_idx], 
                          out_channels=conv_input_channels[layer_idx + 1], 
                          kernel_size=conv_size,
                          stride=stride_size, 
                          padding=padding_size)
                )
    
        # Pooling
        if (layer_idx + 1) <= pooling_layers_max_num:
            # if add pooling layer            
            layers.append(nn.MaxPool2d(pooling_kernel_size, stride=2))
            actual_pooling_layer_num += 1
        # print('pooling')
        
        # Batchnorm
        if batchnorm:
            if layer_idx != n_conv_layers - 1 or batchnorm_at_end:
                # if it's not the last layer or it's forced to be added after the last layer
                layers.append(nn.BatchNorm2d(conv_input_channels[layer_idx + 1]))
        # print('batchnorm')
        # Activation
        if layer_idx != n_conv_layers - 1 or activation_at_end:
            layers.append(activation_func)
        # print('activation')
    if input_H is not None and input_W is not None:
        output_H = input_H // (2**actual_pooling_layer_num)
        output_W = input_W // (2**actual_pooling_layer_num)
    else:
        output_H, output_W = None, None
    
    # ic(layers)
    # ic([type(layer) for layer in layers if type(layers) is nn.Conv2d])
    output_channel_num = [layer for layer in layers if type(layer) == nn.Conv2d][-1].out_channels
    
    # print('WWW')
    return {'layers': layers,
            'output_shape': (output_channel_num, output_H, output_W)}
    # return layers

# def fcn(n_linear_layers, linear_input_feature_dim, linear_inner_feature_dim, linear_output_feature_dim, actiFunc,
#         useBN=False):
#     if n_linear_layers > 1:
#         linear_layers = [nn.Flatten(),
#                          nn.Linear(linear_input_feature_dim, linear_inner_feature_dim),
#                          actiFunc]
#         if useBN:
#             linear_layers += [nn.BatchNorm1d(linear_inner_feature_dim)]
    
#         for inner_layer_idx in range(n_linear_layers - 2):
#             linear_layers += [nn.Linear(linear_inner_feature_dim, linear_inner_feature_dim), actiFunc]
#             if useBN:
#                 linear_layers += [nn.BatchNorm1d(linear_inner_feature_dim)]
    
#         linear_layers += [nn.Linear(linear_inner_feature_dim, linear_output_feature_dim)]
#     else:
#         linear_layers = [nn.Flatten(),
#                          nn.Linear(linear_input_feature_dim, linear_output_feature_dim),
#                          actiFunc]
#         if useBN:
#             linear_layers += [nn.BatchNorm1d(linear_output_feature_dim)]
#     return linear_layers

def fcn(n_linear_layers, 
        linear_input_feature_dim, 
        linear_inner_feature_dim, 
        linear_output_feature_dim, 
        activation_func,
        batchnorm=False, 
        simple_last_layer=True):
    
    if n_linear_layers <=0:
        return {'layers': [], 'output_shape': linear_input_feature_dim}
    
    if n_linear_layers > 1:
        layers = [nn.Flatten()]
        
        
        if linear_inner_feature_dim is None:
            layer_input_dims = np.linspace(linear_input_feature_dim, linear_output_feature_dim, n_linear_layers + 1).astype(int)
        elif type(linear_inner_feature_dim) is int:
            layer_input_dims = [linear_input_feature_dim] + [linear_inner_feature_dim]*(n_linear_layers-2+1) + [linear_output_feature_dim]
        
        # ic(layer_input_dims)                                
        for layer_idx in range(n_linear_layers):
            curr_layer_input_dim = layer_input_dims[layer_idx]
            curr_layer_output_dim = layer_input_dims[layer_idx + 1]
                        
            layers.append(nn.Linear(curr_layer_input_dim, curr_layer_output_dim))
            
            if layer_idx != n_linear_layers-1 or not simple_last_layer:
                layers.append(activation_func)
                if batchnorm:
                    layers.append(nn.BatchNorm1d(curr_layer_output_dim))               
    else:
        layers = [nn.Flatten(),
                         nn.Linear(linear_input_feature_dim, linear_output_feature_dim),
                         activation_func]
        if batchnorm:
            layers += [nn.BatchNorm1d(linear_output_feature_dim)]
    # return layers
    return {'layers': layers}

def conv_fc_net(
        conv_input_shape: list or tuple,
        linear_output_dim:int, 
        conv_layer_num=3,
        conv_output_channel_num:int or None=None,
        conv_kernel_size=3,
        conv_stride_size=1, 
        conv_padding_size=1, 
        conv_pooling_method=None, 
        conv_pooling_kernel_size=2,
        conv_pooling_layers_max_num=None,
        activation_func=torch.nn.ReLU(), 
        batchnorm=False, 
        linear_layer_num=3,        
        linear_inner_dim = None,
        simple_last_linear_layer=True
        ):
    
    # print('conv')
    conv_net = conv2d_net(
        input_shape = conv_input_shape, 
        n_conv_layers = conv_layer_num, 
        n_output_channels = conv_output_channel_num,
        conv_size = conv_kernel_size,
        stride_size = conv_stride_size, 
        padding_size = conv_padding_size,                  
        pooling_method = conv_pooling_method, 
        pooling_kernel_size = conv_pooling_kernel_size,
        pooling_layers_max_num = conv_pooling_layers_max_num,
        activation_func = activation_func, 
        batchnorm = batchnorm,
        batchnorm_at_end = True, 
        activation_at_end = True)
    # print('linear')
    # print(linear_layer_num, np.prod(conv_net['output_shape']), linear_inner_dim, linear_output_dim)
    linear_net = fcn(
        n_linear_layers = linear_layer_num, 
        linear_input_feature_dim = np.prod(conv_net['output_shape']), 
        linear_inner_feature_dim = linear_inner_dim,
        linear_output_feature_dim = linear_output_dim, 
        activation_func = activation_func, 
        batchnorm = batchnorm,
        simple_last_layer = True)
    
    # return conv_net['layers'] + linear_net['layers']
    return {'layers': conv_net['layers'] + linear_net['layers'], 'output_shape': linear_output_dim}

class View(nn.Module):
    """ Custom reshape layer """
    def __init__(self, target_shape):
        # https://github.com/pytorch/vision/issues/720
        super().__init__()   
        # self.shift = shift
        self.target_shape = target_shape

    @autocast()
    def forward(self, x):        
        # x = x.reshape(x.shape[0], self.n_classes, -1)
        # return x.view(self.target_shape)
        return x.reshape(self.target_shape)

class shift_Leaky_ReLU(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, shift):
        super().__init__()   
        self.shift = shift
    
    @autocast()
    def forward(self, x):        
        return nn.LeakyReLU(inplace=True)(x - self.shift) + self.shift